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

!pip install GetOldTweets3
import GetOldTweets3 as got
text_query = 'Coronavirus+India+Hyderabad'
count = 2000
# Creation of query object
tweetCriteria = got.manager.TweetCriteria().setQuerySearch(text_query)\
                                            .setMaxTweets(count)
# Creation of list that contains all tweets
tweets = got.manager.TweetManager.getTweets(tweetCriteria)
# Creating list of chosen tweet data
text_tweets = [[tweet.date, tweet.text,tweet.username,tweet.geo] for tweet in tweets]

