
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
