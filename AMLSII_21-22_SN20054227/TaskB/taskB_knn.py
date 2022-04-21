# this file is for model training and testing
# use tf-idf to convert text.
# use Random Forest, SVM and Logistic Regression to train the model and test.

# import
import re
import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier


df = pd.read_csv('DataCleanB_Eng.csv')  # English Data
# df = pd.read_csv('DataCleanB_Arabic.csv')  # Arabi Data

def tf_idf(contents):
  # word-> matrix: a[i][j] is word frequency of j under text i
  vectorizer = CountVectorizer()

  # Count tf-idf weights for each word
  transformer = TfidfTransformer()

  #fit_transform: tf-idf 
  tfidf = transformer.fit_transform(vectorizer.fit_transform(contents)) # fit_transform: convert text into word-frequency matrix

  # Get all phrases in bag-of-words model 
  word = vectorizer.get_feature_names()
  # print("Length of words:", len(word))
  X = coo_matrix(tfidf, dtype=np.float32).toarray() 


  return X

def KNN_classifier(X_train, X_test, x_val, y_val, y_train, y_test):
  # KNN
  LR = KNeighborsClassifier(n_neighbors=4)
  LR.fit(X_train, y_train)
  val_acc = LR.score(x_val, y_val)
  test_val = LR.score(X_test, y_test)
  train_acc = LR.score(X_train, y_train)
  # print('Validation Accuracy:{}'.format(val_acc))
  # print('Test Accuracy:{}'.format(test_val))
  # print('Train Accuracy:{}'.format(train_acc))
  pre = LR.predict(X_test)
  # print(classification_report(y_test, pre))
  return LR, val_acc, test_val, train_acc

def train_model(df):

  labels = df["Label"].to_list()
  contents = df["Text"].to_list()

  # td-idf
  tf_matrix = tf_idf(contents)

  # split data
  X_train, X_test, y_train, y_test = train_test_split(tf_matrix, labels, test_size=0.4, random_state=1)
  x_val = X_test[10:]
  y_val = y_test[10:]
  x_val = X_test[:10]
  y_val = y_test[:10]

  # logistic regression
  LR, val_acc, test_val, train_acc = KNN_classifier(X_train, X_test, x_val, y_val, y_train, y_test)

  return val_acc, test_val, train_acc