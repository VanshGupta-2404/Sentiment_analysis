
from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import pandas as pd
#import flask as Flask
from flask import Flask,render_template

file_path = 'drive/My Drive/content/Colab Notebooks/Restaurant_Reviews.tsv'

# Read the TSV file
data = pd.read_csv(file_path, delimiter='\t', quoting=3)

# Now you can work with the 'data' DataFrame

data.shape

data.columns

data.head()

data.info

import nltk
import re
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []  # Initialize an empty list to store the cleaned text data

for i in range(0, 1000):  # Loop through the first 1000 rows (adjust as needed)
    review = re.sub(pattern='[^a-zA-Z]', repl=' ', string=data['Review'][i])
    review = review.lower()
    review_words = review.split()
    review_words = [word for word in review_words if not word in set(stopwords.words('english'))]
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review_words]
    review = ' '.join(review)
    corpus.append(review)

corpus[:100]

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer with max_features set to 1500
tfidf_vectorizer = TfidfVectorizer(max_features=1500)

# Fit and transform your 'corpus' text data using TF-IDF
X = tfidf_vectorizer.fit_transform(corpus).toarray()

# Assuming 'data.iloc[:, 1]' contains your target variable
y = data.iloc[:, 1].values

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=0)

X_train.shape, X_test.shape, y_train.shape, y_test.shape

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier
classifier = RandomForestClassifier(n_estimators=100, random_state=0)  # You can adjust the number of estimators as needed

# Fit the classifier to your training data
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
score1= accuracy_score(y_test,y_pred)
score2= precision_score(y_test,y_pred)
score3= recall_score(y_test,y_pred)
print("''''Score'''")
print("Accuracy score is: {}%".format(round(score1*100,2)))
print("Precision score is: {}%".format(round(score2*100,2)))
print("Recall score is: {}%".format(round(score3*100,2)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

cm

# Commented out IPython magic to ensure Python compatibility.
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

plt.figure(figsize= (10,6))
sns.heatmap(cm, annot=True, cmap = "YlGnBu", xticklabels = ['Negative', 'Positive'], yticklabels = ['Negative', 'Positive'])
plt.xlabel('Predict values')
plt.ylabel('Actual value')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

best_accuracy = 0.0
best_n_estimators = 0

for n_estimators in range(1, 101,1):  # Adjust the range of n_estimators as needed
    temp_classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=0)
    temp_classifier.fit(X_train, y_train)
    temp_y_pred = temp_classifier.predict(X_test)
    score = accuracy_score(y_test, temp_y_pred)
    print("Accuracy score for n_estimators={} is: {}%".format(n_estimators, round(score * 100, 2)))

    if score > best_accuracy:
        best_accuracy = score
        best_n_estimators = n_estimators
        print("--------------------------")
        print("The best Accuracy is {}% with n_estimators value as {}".format(round(best_accuracy * 100, 2), best_n_estimators))

from sklearn.ensemble import RandomForestClassifier

# Initialize the Random Forest classifier with a specific number of estimators (trees)
n_estimators = 100  # Adjust the number of trees as needed
classifier = RandomForestClassifier(n_estimators=n_estimators, random_state=0)

# Fit the classifier to your training data
classifier.fit(X_train, y_train)

from flask import Flask
from flask_ngrok import run_with_ngrok

app = Flask(__name__)
run_with_ngrok(app)
@app.route('/')
def part3():
    return '<h1>Welcome to CID</h1>'
app.run()



def predict_sentiment(sample_review):
  sample_review = re.sub(pattern='[^a-zA-Z]',repl = ' ', string = sample_review)
  sample_review = sample_review.lower()
  sample_review_words = sample_review.split()
  sample_review_words = [word for word in sample_review_words if not word in set(stopwords.words('english'))]
  ps = PorterStemmer()
  final_review = [ps.stem(word) for word in sample_review_words]
  final_review = ' '.join(final_review)

  temp = tfidf_vectorizer.transform([final_review]).toarray()
  return classifier.predict(temp)

sample_review = 'The food is really bad'

if predict_sentiment(sample_review):
  print("This is a positive review")
else:
  print('This is Negative review')

sample_review = 'The food is great'

if predict_sentiment(sample_review):
  print("This is a positive review")
else:
  print('This is Negative review')

sample_review = 'The amazing dine'

if predict_sentiment(sample_review):
  print("This is a positive review")
else:
  print('This is Negative review')

sample_review = 'I appreciate the Restraunt efforts,  i liked the food, feedback from my side is to make food more spicy, also reduce service charge'

if predict_sentiment(sample_review):
  print("This is a positive review")
else:
  print('This is Negative review')

if __name__ == "__main__":
    app.run(debug=True)
