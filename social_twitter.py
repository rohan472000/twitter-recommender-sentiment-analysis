import tweepy
import pandas as pd

# Authenticate with Twitter API
import os

access_key = os.getenv("ACCESS_KEY")
access_secret = os.getenv("ACCESS_SECRET")
consumer_key = os.getenv("CONSUMER_KEY")
consumer_secret = os.getenv("CONSUMER_SECRET")

# Twitter authentication
auth = tweepy.OAuthHandler(access_key, access_secret)
auth.set_access_token(consumer_key, consumer_secret)

# # # Creating an API object
api = tweepy.API(auth)

import tweepy
# from tweepy import OAuthHandler
import pandas as pd



# import s3fs


def run_twitter_etl():
    tweets = api.user_timeline(screen_name='@shakira',
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


# run_twitter_etl()# uncomment if needed

import pandas as pd

data = pd.read_csv('reftweets.csv')
len(data)

import pandas as pd
from textblob import TextBlob

# Load data
data = pd.read_csv('reftweets.csv')

# Convert 'text' column to string type
data['text'] = data['text'].astype(str)

# Perform sentiment analysis on each tweet
sentiment = []
scale = []
for tweet in data['text']:
    analysis = TextBlob(tweet)
    polarity = analysis.sentiment.polarity
    if polarity > 0:
        sentiment.append('positive')
        scale.append(1)
    elif polarity < 0:
        sentiment.append('negative')
        scale.append(-1)
    else:
        sentiment.append('neutral')
        scale.append(0)

# Add sentiment column to data
data['sentiment'] = sentiment
data['scale'] = scale
# Save updated data
# data.to_csv('updated_reftweets.csv', index=False) # uncomment if needed

# data1 = pd.read_csv('/content/updated_reftweets.csv')


import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# import seaborn as sns
import matplotlib.pyplot as plt

# Load sentiment analysis data
data = pd.read_csv('updated_reftweets.csv')

# Preprocess data
X = data['text'].astype(str)
y = data['sentiment']

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM model
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Make predictions on test set
y_pred = clf.predict(X_test)

# Evaluate model using various metrics
accuracy = accuracy_score(y_test, y_pred)
# confusion_mtx = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Print results
print('Accuracy:', accuracy)
print('Classification Report:\n', report)

import re
import tweepy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Define function to preprocess tweets
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
# input_user = input("Enter the username to get recommendations: ")
# user_tweets = get_user_tweets(api, ('@' + input_user))
user_tweets = get_user_tweets(api, '@_4_anand_')
test_tweets = user_tweets
print(user_tweets)
# Preprocess and vectorize user's tweets
user_tweets = [preprocess_tweet_one(tweet) for tweet in user_tweets]
test_tweets1 = user_tweets
user_tweets = vectorizer.transform(user_tweets)

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
# ------------------------------------------------------------------------------------
# Convert 'user' column to string data type
data['user'] = data['user'].astype(str)

# Find other users who have tweeted with similar sentiment
similar_userss = data[data['sentiment'] == user_sentiment]['user'].unique()

# Filter out any NaN values from the list of similar users
similar_userss = [user for user in similar_userss if str(user) != 'nan']
print('similar users are : ', similar_userss)
# ----------------------------------------------------------------------------------------
import pandas as pd
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import numpy as np

# Load the training data
data = pd.read_csv('updated_reftweets.csv')
print(data['user'].unique())
# drop rows with usernames '3124', 'nan', and '7329'
data = data.drop(data[data['user'].isin(['3124', np.nan, '7329', 'user'])].index)

# print the updated DataFrame
print(data['user'].unique())


# Preprocess tweet text by removing URLs, mentions, and special characters
def preprocess_tweet(tweet):
    # Remove URLs
    tweet = re.sub(r'http\S+', '', tweet)
    # Remove mentions
    tweet = re.sub(r'@\w+', '', tweet)
    # Remove special characters and digits
    tweet = re.sub(r'[^\w\s]', '', tweet)
    tweet = re.sub(r'\d+', '', tweet)
    # Convert to lowercase
    tweet = tweet.lower()
    return tweet


# Convert 'user' column to string data type
data['user'] = data['user'].astype(str)

# Train a classifier to predict sentiment from text
vectorizer = CountVectorizer(preprocessor=preprocess_tweet)
X = vectorizer.fit_transform(data['text'])
y = data['sentiment']
clf = MultinomialNB().fit(X, y)

# Set a threshold for similarity
similarity_threshold = 0.5


# Calculate similarity score between two strings
def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


# Function to assign weight to each user based on their scale value
def get_user_weight(user_scale, tweet_scale):
    if user_scale == tweet_scale:
        return 1.0
    elif user_scale == 0 or tweet_scale == 0:
        return 0.5
    else:
        return 0.0


# Get input tweet data
# assign_df = pd.read_csv('input_tweets.csv')

# Remove rows with null/nan values
assign_df = assign_df.dropna()

# Iterate over input tweets
for index, tweet in assign_df.iterrows():
    # Preprocess the input tweet
    test_tweet = preprocess_tweet(tweet['text'])
    # Calculate sentiment polarity score for the input tweet
    tweet_sentiment = clf.predict(vectorizer.transform([test_tweet]))[0]
    # Get scale value for the input tweet
    tweet_scale = tweet['scale']
    # Find other users who have tweeted with similar sentiment
    similar_users_tweet = []
    for user in data['user'].unique():
        if str(user) != 'nan':
            user_sentiment = clf.predict(vectorizer.transform(data[data['user'] == user]['text']))[0]
            user_scale = data[data['user'] == user]['scale'].values[0]
            # Check if user sentiment is similar to tweet sentiment
            if similarity(user_sentiment, tweet_sentiment) > similarity_threshold:
                # Assign weight to user based on scale value
                weight = get_user_weight(user_scale, tweet_scale)
                if weight > 0:
                    similar_users_tweet.append((user, weight))
    # Sort the list of similar users by weight in descending order
    similar_users_tweet = sorted(similar_users_tweet, key=lambda x: x[1], reverse=True)
    # Print the top 10 recommended users for the input tweet
    print(f"Recommended users for tweet {index + 1}:{tweet}")
    for user, weight in similar_users_tweet[:10]:
        print(f"{user} ({weight:.2f})")
    print()


def recommend_users_for_user_final(username):
    # Preprocess the username
    username = preprocess_tweet(username)
    # Find the scale value for the user
    user_scale = data[data['user'] == username]['scale'].unique()
    # Find other tweets with similar sentiment from the user
    similar_tweets = data[(data['user'] == username) & (data['sentiment'] == tweet_sentiment)]['text']
    # Combine the similar tweets into one text
    similar_tweet_text = ' '.join(similar_tweets)
    # Calculate sentiment polarity score for the similar tweets
    similar_tweet_sentiment = clf.predict(vectorizer.transform([preprocess_tweet(similar_tweet_text)]))[0]
    # Find other users who have tweeted with similar sentiment
    similar_users = []
    for user in data['user'].unique():
        if str(user) != 'nan' and user != username:
            user_sentiment = clf.predict(vectorizer.transform(data[data['user'] == user]['text']))[0]
            user_scale = data[data['user'] == user]['scale'].values[0]
            # Check if user sentiment is similar to tweet sentiment
            if similarity(user_sentiment, similar_tweet_sentiment) > similarity_threshold:
                # Assign weight to user based on scale value
                weight = get_user_weight(user_scale, user_scale)
                if weight > 0:
                    similar_users.append((user, weight))
    # Sort the list of similar users by weight in descending order
    similar_users = sorted(similar_users, key=lambda x: x[1], reverse=True)
    # Return the top recommended users
    return [user for user, weight in similar_users[:10]]


# input_name = input("Enter the username to get last ------- recommendations: ")
# username = '@' + input_name
# recommended_users = recommend_users_for_user_final(username)
# print(f"Recommended users for {username}: {recommended_users}")
