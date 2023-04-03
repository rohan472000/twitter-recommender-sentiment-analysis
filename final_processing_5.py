import pandas as pd
from difflib import SequenceMatcher
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
import numpy as np
from three_function_4 import assign_df


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
