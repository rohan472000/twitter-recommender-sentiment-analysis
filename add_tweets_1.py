import os

# Authenticate with Twitter API
import tweepy
import pandas as pd

KEYS = "ACCESS_KEY ACCESS_SECRET CONSUMER_KEY CONSUMER_SECRET".split()
assert all(key in os.environ for key in KEYS)
assert not any(os.getenv(key) for key in KEYS)

# Twitter authentication
auth = tweepy.OAuthHandler(os.getenv("ACCESS_KEY"), os.getenv("ACCESS_SECRET"))
auth.set_access_token(os.getenv("CONSUMER_KEY"), os.getenv("CONSUMER_SECRET"))

# # # Creating an API object
api = tweepy.API(auth)

import pandas as pd


def run_twitter_etl():
    input_name = input("Enter the username to add his/her tweets: ")
    username = '@' + input_name
    tweets = api.user_timeline(screen_name=username,
                               # 200 is the maximum allowed count
                               count=200,
                               include_rts=False,
                               # Necessary to keep full_text
                               # otherwise only the first 140 words are extracted
                               tweet_mode='extended'
                               )

    list = []
    for tweet in tweets:
        text = tweet._json["full_text"]

        refined_tweet = {"user": tweet.user.screen_name,
                         'text': text,
                         'favorite_count': tweet.favorite_count,
                         'retweet_count': tweet.retweet_count,
                         'created_at': tweet.created_at}

        list.append(refined_tweet)

    df = pd.DataFrame(list)
    df.to_csv('reftweets.csv', mode='a')

# run_twitter_etl()
