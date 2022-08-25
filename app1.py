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
from __future__ import absolute_import, print_function
import time
import json
import datetime
import pytz
from datetime import datetime
from datetime import timedelta

# Import modules
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import dataset
from sqlalchemy.exc import ProgrammingError

app = Flask(__name__,static_url_path='/static')
Bootstrap(app)
mysql = MySQL()
app.config['MYSQL_DATABASE_USER'] = 'root'
app.config['MYSQL_DATABASE_PASSWORD'] = ''
app.config['MYSQL_DATABASE_DB'] = 'suicidewatch'
app.config['MYSQL_DATABASE_HOST'] = 'localhost'
mysql.init_app(app)



@app.route('/index')
def index():
	eat = pytz.timezone('Africa/Nairobi')
	fmt = '%Y-%m-%d %H:%M:%S'
	
	conn = mysql.connect()
	cursor =conn.cursor()

	
	import re # importing regex
	import string
	import tweepy
	
	consumer_key = "NCDRIUJUg2fPO2wC4Uv7klSd7"
	consumer_secret = "QqjcDyJw7jTIS4mCw3oM5lZn7GIMP7jeUi2gEc3lhPZJtNUAEZ"
	access_token = "723018819111809024-6EnoCpbcJEgk9yO6HwbfrCYb5oQHcra"
	access_token_secret = "p3BTD7On6A2p833EfQ91Nth81fcQPIrSsf0odF4eHTfK4"
	
	class StdOutListener(StreamListener):
    """ A listener handles tweets that are received from the stream.
    This is a basic listener that just prints received tweets to stdout.
    """
    def on_status(self, status):
        
        
        print(status.text)        
        print(status.user.screen_name)
        print(status.user.name)        
        ptime = (status.created_at)+timedelta(hours=3)       
        print(ptime)
        
        return True

    def on_error(self, status_code):
        if status_code == 420:
            return False
	

	auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
	auth.set_access_token(access_token, access_token_secret)
	api = tweepy.API(auth)
	tweets = api.user_timeline('@d_kurui', count=100, tweet_mode='extended')
	
	
	tag_map = defaultdict(lambda : wn.NOUN)
	tag_map['J'] = wn.ADJ
	tag_map['V'] = wn.VERB
	tag_map['R'] = wn.ADV



	# Step - 4: Vectorize the words by using TF-IDF Vectorizer - This is done to find how important a word in document is in comaprison to the corpus
	
	def list_tweets(user_id, count, prt=False):
		tweets = api.user_timeline("@" + user_id, count=count, tweet_mode='extended')
		tw = []
		for t in tweets:
			tw.append(t.full_text)
			if prt:
				print(t.full_text)
				print()
		return tw
	
	user_id = 'd_kurui'
	count=1
	mytweets = list_tweets(user_id, count)

	def remove_pattern(input_txt, pattern):
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
		
	def clean_tweet(tweet):
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
	
	def pre_process_tweet(incomingtweet):
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
	cleantweets = []
	for i in range(len(mytweets)):
		cleantweets.append(mytweets[i])
		#print(clean_tweet(mytweets[i]))  
		
	mymodel = pickle.load(open('models/suicide_final_model.pkl','rb'))
	from datetime import datetime
	today = datetime.today().strftime('%Y-%m-%d')
	for i in range(len(cleantweets)):
		print(cleantweets[i])
		twoutput = (pre_process_tweet(cleantweets[i]))      
		output = mymodel.predict(twoutput)
		print(output)
  
		print("-----------------------") 
		
		if output == 1:        
			print("save")
			tweet_insert = "INSERT  INTO tweets(username,dateposted,tweet,attendedto) VALUES(%s,%s,%s,%s)"		
			cursor.execute(tweet_insert, (user_id,today,cleantweets[i],0))
			conn.commit()
			#print(user_id)
			#print(cleantweets[i])
		elif output == 0:
			print("Dont save")
			print(user_id)
			print(cleantweets[i])	
			
			
	
	select_tweets = "SELECT username, dateposted, tweet, attendedto FROM tweets"
	cursor.execute(select_tweets)
	retrieved_tweets = cursor.fetchall()
	
	
	
	return render_template('index1.html',data = retrieved_tweets)
	
if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5000, debug=True)
	l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    stream = Stream(auth, l)
    #stream.filter(track=['Strathmore', 'sofapaka'])
    stream.filter(follow=['723018819111809024','609884475'])


