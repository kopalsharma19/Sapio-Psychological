# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 13:27:29 2020

@author: ADMIN
"""

import pandas as pd
import time

consumer_key = "HiykXObIJpZfcyYTsP7gMppck"
consumer_secret = "Pk2Pi2NQOZzGIgSRKiHvvf51QNpOZXgUIDC8YEDkC5k2ZqNplc"
access_token = "1224697451275223040-w0EfkaGfnrgS1FFjzmpe1BFZiVq6Fp"
access_token_secret = "FnXQ8IbEas2YtIpL24vpkAqjTVvNDrsJxWgtWJRW6jKoP"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit=True)
import datetime
!pip install GetOldTweets3
import GetOldTweets3 as got
    text_query = 'covid+India'
    count = 10000
    #sdate = datetime.datetime(2020, 5, 1) 
    # Creation of query object
    tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query)\
                                                .setSince("2020-03-14")\
                                               .setUntil("2020-03-20")\
                                               .setMaxTweets(count)
                                               
    # Creation of list that contains all tweets
    tweets = got.manager.TweetManager.getTweets(tweetCriteria)
    # Creating list of chosen tweet data
    text_tweets1 = [[tweet.date, tweet.text,tweet.username] for tweet in tweets]   

df=pd.DataFrame("Tweets_before.csv")
'
df.to_csv('Tweets_before.csv')
