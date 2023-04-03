# Iterate over input tweets
from final_processing_5 import assign_df, preprocess_tweet, clf, vectorizer, data, similarity, \
    similarity_threshold, get_user_weight

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
    # print(f"Recommended users for tweet {index + 1}:{tweet}")
    # for user, weight in similar_users_tweet[:10]:
    #     print(f"{user} ({weight:.2f})")
    # print()


def recommend_users_for_user_final(username):
    # Preprocess the username
    username = preprocess_tweet(username)
    # Find the scale value for the user
    # user_scale = data[data['user'] == username]['scale'].unique()
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