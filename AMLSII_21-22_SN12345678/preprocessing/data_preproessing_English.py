# this file is for data prepocessing:
# clean the original Twitter texts

# import
import re
import pandas as pd 
import numpy as np 
from wordcloud import WordCloud
from google.colab import drive
from collections import Counter
import matplotlib.pyplot as plt

df = pd.read_csv('twitterA_train_data.txt', sep="\t",names=['Number','Label','Text'])


def remove_handles(input):
  for i in re.findall("@[\w]*|#[\w]*", input):
    input = re.sub(i, '', input)
  # for i in re.findall("#[\w]*", input):
  #   input = re.sub(i, '', input)
  return input


def data_prepocessing(df):
  # remove twitter sentences in @xxx or #xxx format
  df['handles_removed'] = np.vectorize(remove_handles)(df['Text'])

  # remove special characters, numbers, punctuations
  df['spec_char_removed'] = df['handles_removed'].str.replace("[^a-zA-Z#]", " ")

  # remove world with length<=2
  df['tidy'] = df['spec_char_removed'].apply(lambda x: ' '.join([i for i in x.split(" ") if len(i) > 2]))

  return df


# data exploration
def data_exploration(df):
  # Labels:
  print("Label Distribuition")
  counter = Counter(df.Label)
  plt.figure(figsize=(5, 3))
  plt.bar(counter.keys(), counter.values())
  print(counter.keys())
  print(counter.values())
  plt.title("Labels Distribuition")

  # word cloud
  all_words = ' '.join([text for text in df['tidy']])
  wordcloud = WordCloud(width=300, height=200, random_state=21, max_font_size=50).generate(all_words)
  plt.figure(figsize=(8, 5))
  plt.imshow(wordcloud)
  plt.imshow(wordcloud, interpolation="bilinear")
  plt.show()

  # max length of twitter
  documents = [_text.split() for _text in df.Text]
  max_len = 0
  for _text in df.Text:
    token = [t for t in _text.split()]
    max_len = len(token) if max_len > len(token) else max_len
  # print("Max Twitter Length:")
  # print(max_len)