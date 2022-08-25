#!/usr/bin/env python
# coding: utf-8

# In[39]:


from __future__ import absolute_import, print_function
from __future__ import print_function
import time
import json
import datetime
import pytz
from datetime import datetime
from datetime import timedelta
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd 
from flask import Flask, render_template, url_for,request, jsonify
from flask_bootstrap import Bootstrap 
from sklearn.ensemble import RandomForestRegressor
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from flaskext.mysql import MySQL
import pymysql
from matplotlib import pyplot as plt
from scipy.interpolate import spline
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from io import BytesIO
from flask import Flask, render_template, send_file, make_response
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import africastalking
import numpy as np

from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pymysql



pymysql.install_as_MySQLdb()

import time
import json
import datetime
import pytz
from datetime import datetime
from datetime import timedelta

eat = pytz.timezone('Africa/Nairobi')
fmt = '%Y-%m-%d %H:%M:%S'

# Import modules
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import dataset
from sqlalchemy.exc import ProgrammingError

# Your credentials go here
consumer_key = "NCDRIUJUg2fPO2wC4Uv7klSd7"
consumer_secret = "QqjcDyJw7jTIS4mCw3oM5lZn7GIMP7jeUi2gEc3lhPZJtNUAEZ"
access_token = "723018819111809024-6EnoCpbcJEgk9yO6HwbfrCYb5oQHcra"
access_token_secret = "p3BTD7On6A2p833EfQ91Nth81fcQPIrSsf0odF4eHTfK4"


tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

import MySQLdb

conn = MySQLdb.connect("localhost","root","","suicidewatch")
c = conn.cursor()


"""
The code for our listener class above goes here!


"""


class SMS:
	def __init__(self):
		# Set your app credentials
		self.username = "dkurui"
		self.api_key = "3f8af3d9094245072eaccf278d21bc2c9e8ace346fb4540c8bcc653c406a1785"
		
        # Initialize the SDK
		africastalking.initialize(self.username, self.api_key)

        # Get the SMS service
		self.sms = africastalking.SMS
	def send(self, msg):
            # Set the numbers you want to send to in international format
            recipients = ["+254722680993"]

            # Set your message
            message = msg
			#"I'm a lumberjack and it's ok, I sleep all night and I work all day";

            # Set your shortCode or senderId
            #sender = "SUICIDE ALERT"
            try:
				# Thats it, hit send and we'll take care of the rest.
                response = self.sms.send(message, recipients)
                print (response)
            except Exception as e:
                print ('Encountered an error while sending: %s' % str(e))


class StdOutListener(StreamListener):
	
  
	def remove_pattern(self,input_txt, pattern):
		r = re.findall(pattern, input_txt)
		for i in r:
			input_txt = re.sub(i, '', input_txt)        
		return input_txt
		
	def clean_tweets(lst):
		# remove twitter Return handles (RT @xxx:)
		lst = np.vectorize(remove_pattern)(lst, "RT @[\w]*:")
		# remove twitter handles (@xxx)
		lst = np.vectorize(remove_pattern)(lst, "@[\w]*")
		# remove URL links (httpxxx)
		lst = np.vectorize(remove_pattern)(lst, "https?://[A-Za-z0-9./]*")
		# remove special characters, numbers, punctuations (except for #)
		lst = np.core.defchararray.replace(lst, "[^a-zA-Z#]", " ")
		return lst
		
	def clean_tweet(self,tweet):
		# remove old style retweet text "RT"
		tweet = re.sub(r'^RT[\s]+', '', tweet) 
		# remove hyperlink
		tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet) 
		# remove hashtags
		# only removing the hash # sign from the word
		tweet = re.sub(r'#', '', tweet)
		# remove mentions
		tweet = re.sub(r'@[A-Za-z0-9]+', '', tweet)  
		# remove punctuations like quote, exclamation sign, etc.
		# we replace them with a space
		tweet = re.sub(r'['+string.punctuation+']+', ' ', tweet)
		return tweet 
	
	def pre_process_tweet(self,incomingtweet):
		lowertest = [incomingtweet.lower()]
		#tokenize
		newlowest= [word_tokenize(entry) for entry in lowertest]
		series=pd.Series(newlowest).astype(str)

		for index,entry in enumerate(newlowest):
			# Declaring Empty List to store the words that follow the rules for this step
			Final_words = []
			# Initializing WordNetLemmatizer()
			word_Lemmatized = WordNetLemmatizer()
			# pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
			for word, tag in pos_tag(entry):
				# Below condition is to check for Stop words and consider only alphabets
				if word not in stopwords.words('english') and word.isalpha():
					word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
					Final_words.append(word_Final)
			# The final processed set of words for each iteration will be stored in 'text_final'
			#newlowest.loc[index,'text_final'] = str(Final_words)
			lseries=str(Final_words)
			
		series=pd.Series(lseries).astype(str)  
		pl = pickle.load(open('models/tfidf.pickle','rb'))		   
		FinalTweet = pl.transform(series) 		
		return FinalTweet	

    
	def on_status(self,status):		
		ptime = (status.created_at)+timedelta(hours=3)
		incomingtw = status.text
		clean_tweet = l.pre_process_tweet(incomingtw)
		mymodel = pickle.load(open('models/suicide_final_model.pkl','rb'))
		output = mymodel.predict(clean_tweet)
		
		
		if output == 1:        
			print("saving into DB")			
			tweet_insert = "INSERT  INTO tweets(username,screenname,dateposted,tweet,attendedto) VALUES(%s,%s,%s,%s,%s)"	
			attendedto = 'NO'
			c.execute(tweet_insert, (status.user.name,status.user.screen_name,ptime,status.text,attendedto))			
			conn.commit()			
			SMS().send(status.text)	
			
						
		elif output == 0:
			print("Dont save")
			print(status.user.screen_name)
			print(status.text)	
			print("--------")
			
		return True
		
		
	def on_error(self, status_code):
		if status_code == 420:
			return False       
        

if __name__ == '__main__':	
	l = StdOutListener()
	auth = OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	stream = Stream(auth, l)
	#stream.filter(track=['Strathmore', 'sofapaka'])
	stream.filter(follow=['723018819111809024','609884475','3101553649', '1123655390938450000', '1074329919990230000', '4292999657', '2362631725', '4075102865', '800278627', '200672890', '535423827', '1098514176257990000', '1096507551003430000', '1091600406034230000', '3903589872', '617196939', '705618833789362000', '714150447251136000', '114396856', '1399794822', '3420110889', '142940979', '2515175300', '706149320156909000', '1053148392283620000', '731203780537159000', '1910032568', '323951950', '1042398130887690000', '1043469733897620000', '420262979', '3588409096', '384929847', '2744831576', '381444093', '1238947628', '1026231986262860000', '1025786204745330000', '1285318740', '1569810212', '1010460534754430000', '1006771321009820000', '2531575873', '976371711879892000', '972066742443929000', '716285143066013000', '1111066508', '961138401981419000', '2469536199', '960537396768067000', '536241598', '950585174529626000', '582159870', '947728813571694000', '930702986925559000', '924004957443026000', '772537890421309000', '915671294015889000', '500627827', '419709959', '2907433307', '909317899134881000', '875078377711038000', '1392588325', '218468786', '893543320995196000', '3166227413', '881310333830287000', '2249955429', '3006390909', '879291977178763000', '874597712838238000', '869968864733532000', '868903885255192000', '716679125', '868558464041332000', '868507721263349000', '3320842972', '695370579923173000', '863719161779949000', '863106066942488000', '2415796379', '3381181954', '856076558821404000', '848226923704659000', '846675582943481000', '846322902689558000', '2845034392', '829637288338583000', '509946109', '824885902384041000', '366794226','820182718088675000', '800539123', '1550078084', '498639329', '809320929805369000', '728852634250162000', '798443577986715000', '4554121581', '797136225673105000', '3727998681', '2672495485', '1329676591', '2171768991', '726248737', '333978466', '521424488', '791551976303648000', '774968768074088000', '2755661363', '786943394576732000', '528401781', '485052578',  '766712082138013000', '774636050694168000', '256496753', '2713981965', '766350254346297000', '712649580110876000', '3389338408', '756585874423414000', '755507114068574000', '338340785', '751415083339776000', '332189859', '3369065067', '186892047', '1024875522', '734821978469740000', '2653551918', '284107166', '2860123151', '261312570', '3352084187', '376370663', '211748978', '320662182', '729738042093735000', '344575961', '4023551368', '346543467', '411544669', '332966966', '2992501156', '1543283311', '805548007', '239594416', '306121515', '29660100', '248331033','723043107319721000', '352459266', '277378495',])		
	# In[ ]:




