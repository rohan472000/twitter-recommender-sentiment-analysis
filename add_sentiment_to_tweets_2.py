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
data.to_csv('updated_reftweets.csv', index=False) # uncomment if needed
