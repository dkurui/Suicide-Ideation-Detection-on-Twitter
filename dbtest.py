
import MySQLdb
 
print("Connecting to database using MySQLdb")
 
db_connection = MySQLdb.connect(host='localhost',
								db='suicidewatch',
								user='root',
								passwd='')
 
print("Succesfully Connected to database using MySQLdb!")
 
db_connection.close()