from flask import Flask, render_template, url_for,request, jsonify,redirect
#from flask_bootstrap import Bootstrap 
from flaskext.mysql import MySQL
import re
import string

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
	conn = mysql.connect()
	cursor =conn.cursor()
	
	status = "NO"

	select_tweets = "SELECT username, screenname, tweet,dateposted,  attendedto FROM tweets WHERE attendedto = %s"
	cursor.execute(select_tweets, (status))
	retrieved_tweets = cursor.fetchall()
	
	
	return render_template('index1.html',data = retrieved_tweets)
	
@app.route('/attendedto')
def attendedto():
	conn = mysql.connect()
	cursor =conn.cursor()
	
	status = "YES"

	select_tweets = "SELECT username, screenname, tweet,dateposted,  attendedto FROM tweets WHERE attendedto = %s"
	cursor.execute(select_tweets, (status))
	retrieved_tweets = cursor.fetchall()
	
	return render_template("attendedto.html", data = retrieved_tweets)
	
	
@app.route("/nonsuicidal", methods=["POST"])
		
def nonsuicidal():
	tw = request.form.get("twt")
	dp = request.form.get("dp")
	conn = mysql.connect()
	cursor1 =conn.cursor()
	cursor2 =conn.cursor()
	
	delete_tweet = "DELETE FROM tweets WHERE dateposted= %s "	
	cursor1.execute(delete_tweet, (dp))	


	# remove old style retweet text "RT"
	tw = re.sub(r'^RT[\s]+', '', tw) 
	# remove hyperlink
	tw = re.sub(r'https?:\/\/.*[\r\n]*', '', tw) 
	# remove hashtags
	# only removing the hash # sign from the word
	tw = re.sub(r'#', '', tw)
	# remove mentions
	tw = re.sub(r'@[A-Za-z0-9]+', '', tw)  
	# remove punctuations like quote, exclamation sign, etc.
	# we replace them with a space
	tw = re.sub(r'['+string.punctuation+']+', ' ', tw)	
	
	
	final_tweet = tw	
	
	insert_tweet = "INSERT INTO nonsuicidal_tweets (tweet) VALUES (%s)"	
	cursor2.execute(insert_tweet, (final_tweet))	
	conn.commit()

		
	return redirect(url_for("index")) 
	
@app.route("/update", methods=["POST"])
def update():
	tw = request.form.get("twt")
	dp = request.form.get("dp")
	conn = mysql.connect()
	cursor1 =conn.cursor()
	cursor2 =conn.cursor()
	
	update_tweet = "UPDATE tweets SET  attendedto  =  'YES' WHERE dateposted = %s"	
	cursor1.execute(update_tweet, (dp))	
	
	insert_tweet = "INSERT INTO suicidal_tweets (tweet) VALUES (%s)"	
	cursor2.execute(insert_tweet, (tw))
	
	conn.commit()
	
	
	return redirect(url_for("index")) 
	
if __name__ == '__main__':
    app.run(host = '0.0.0.0',port=5000, debug=True)


