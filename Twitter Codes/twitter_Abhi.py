import tweepy
import pandas as pd
import datetime

consumer_key = "HiykXObIJpZfcyYTsP7gMppck"
consumer_secret = "Pk2Pi2NQOZzGIgSRKiHvvf51QNpOZXgUIDC8YEDkC5k2ZqNplc"
access_token = "1224697451275223040-w0EfkaGfnrgS1FFjzmpe1BFZiVq6Fp"
access_token_secret = "FnXQ8IbEas2YtIpL24vpkAqjTVvNDrsJxWgtWJRW6jKoP"
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)

tweets = []
text_query = 'geocode:17.3850,78.4867,30km, covid'

# startDate = datetime.datetime(2020, 5, 10, 0, 0, 0)
# endDate =   datetime.datetime(2020, 5, 29, 0, 0, 0)

print(startDate)

for tweet in api.search(q=text_query, count=100):
  	# Appending chosen tweet data
  	#if tweet.created_at < endDate and tweet.created_at > startDate:
		     tweets.append((tweet.created_at,tweet.text, tweet.user.screen_name))


#print(tweets)

df= pd.DataFrame(tweets, columns=['Date', 'Text', 'User'])
print(df)

df.to_csv('Tweets2.csv')



#print(tweets)