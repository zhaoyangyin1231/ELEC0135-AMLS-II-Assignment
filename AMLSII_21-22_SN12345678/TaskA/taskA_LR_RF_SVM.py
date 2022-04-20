# this file is for model training and testing
# use tf-idf to convert text.
# use Random Forest, SVM and Logistic Regression to train the model and test.

# import
import re
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
import string
import nltk
from wordcloud import WordCloud

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import coo_matrix

from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB


df = pd.read_csv('dataA_tidy.txt', sep="\t",names=['Number','Label','Text'])
df
df["tidy"] = df["Text"]


def tf_idf(contents):
  # word-> matrix: a[i][j] is word frequency of j under text i
  vectorizer = CountVectorizer()

  # Count tf-idf weights for each word
  transformer = TfidfTransformer()

  #fit_transform: tf-idf 
  tfidf = transformer.fit_transform(vectorizer.fit_transform(contents)) # fit_transform: convert text into word-frequency matrix

  # Get all phrases in bag-of-words model 
  word = vectorizer.get_feature_names()
  print("Length of words:", len(word))

  #将tf-idf矩阵抽取出来，元素w[i][j]表示j词在i类文本中的tf-idf权重
  X = coo_matrix(tfidf, dtype=np.float32).toarray() 

  return X


def logistic_reg(X_train, X_test, x_val, y_val, y_train, y_test):
  # LogisticRegression
  LR = LogisticRegression(solver='liblinear')
  LR.fit(X_train, y_train)
  print("LogisticRegression:")
  print('Validation Accuracy:{}'.format(LR.score(x_val, y_val)))
  print('Test Accuracy:{}'.format(LR.score(X_test, y_test)))
  pre = LR.predict(X_test)
  print(classification_report(y_test, pre))
  return LR

def random_forest(X_train, X_test, x_val, y_val, y_train, y_test):
  # RandomForestClassifier
  clf = RandomForestClassifier(n_estimators=20)
  clf.fit(X_train, y_train)
  print("RandomForestClassifier:")
  print('Validation Accuracy:{}'.format(clf.score(x_val, y_val)))
  print('Test Accuracy:{}'.format(clf.score(X_test, y_test)))
  pre = clf.predict(X_test)
  print(classification_report(y_test, pre))
  
def svm_training(X_train, X_test, x_val, y_val, y_train, y_test):
  # SVM classification
  SVM = svm.LinearSVC() #LinearSVC
  SVM.fit(X_train, y_train)
  print("SVMClassifier:")
  print('Validation Accuracy:{}'.format(SVM.score(x_val, y_val)))
  print('Test Accuracy:{}'.format(SVM.score(X_test, y_test)))
  pre = SVM.predict(X_test)
  print(classification_report(y_test, pre))