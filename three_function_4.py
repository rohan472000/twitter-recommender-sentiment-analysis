# Define function to preprocess tweets
import re
import tweepy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from add_tweets_1 import api


def preprocess_tweet_one(tweet):
    # Remove URLs, RTs, and mentions
    tweet = re.sub(r'http\S+|www\S+|https\S+', '', tweet, flags=re.MULTILINE)
    tweet = re.sub(r'\bRT\b', '', tweet)
    tweet = re.sub(r'@\w+', '', tweet)

    # Tokenize tweet and remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(tweet)
    tokens = [token for token in tokens if token.lower() not in stop_words]

    # Stem words
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    # Join tokens back into a string
    tweet = ' '.join(tokens)

    return tweet


# Define function to get user's tweets
def get_user_tweets(api, username):
    tweets = []
    for tweet in tweepy.Cursor(api.user_timeline, id=username, tweet_mode='extended').items():
        if 'retweeted_status' not in dir(tweet):
            tweets.append(preprocess_tweet_one(tweet.full_text))
    return tweets


def assign_scale(df):
    df.loc[df['sentiment'] == 'positive', 'scale'] = 1
    df.loc[df['sentiment'] == 'neutral', 'scale'] = 0
    df.loc[df['sentiment'] == 'negative', 'scale'] = -1
    return df


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

# Load sentiment analysis data
data = pd.read_csv('updated_reftweets.csv')

import nltk

nltk.download('stopwords')
nltk.download('punkt')
data['text'] = data['text'].astype(str)
# Preprocess data
data['text'] = data['text'].apply(preprocess_tweet_one)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['sentiment']

# Train SVM model
clf = SVC(kernel='linear')
clf.fit(X, y)

# Collect user's tweets
input_user = input("Enter the username to get recommendations: ")
user_tweets = get_user_tweets(api, ('@' + input_user))
# user_tweets = get_user_tweets(api, '@_4_anand_')
test_tweets = user_tweets


# print(user_tweets)
# Preprocess and vectorize user's tweets
user_tweets = [preprocess_tweet_one(tweet) for tweet in user_tweets]
test_tweets1 = user_tweets
user_tweets = vectorizer.transform(user_tweets)
# def input_user_tweets_predicting_sentiment(user_tweets):
#     user_tweets = [preprocess_tweet_one(tweet) for tweet in user_tweets]
#     test_tweets1 = user_tweets
#     user_tweets = vectorizer.transform(user_tweets)


# Predict sentiment of user's tweets using the trained SVM model
user_sentiment = clf.predict(user_tweets)

# Add user_sentiment to the data DataFrame
user_df = pd.DataFrame({'text': user_tweets, 'sentiment': user_sentiment})
print(user_df.columns)
user_df1 = pd.DataFrame({'text': test_tweets1, 'sentiment': user_sentiment})
assign_df = assign_scale(user_df1)
data = pd.concat([data, user_df], ignore_index=True)

# Predict sentiment of user's tweets using the trained SVM model
user_sentiment = clf.predict(user_tweets)
print(user_sentiment.shape)

n = user_sentiment.shape[0]

# Slice the data DataFrame to match the length of user_sentiment
data = data[:n]
print('lentgh of data', len(data))
print('lentgh of user_sentiment', len(user_sentiment))
