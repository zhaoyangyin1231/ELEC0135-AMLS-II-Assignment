from data_preproessing_Arabic.py import *
from data_preproessing_English.py import *
from taskA_bidirectional_lstm.py import *
from taskA_LR_RF_SVM.py import *
from taskB_knn.py import *
from taskB_lstm.py import *



# ======================================================================================================================
# Data preprocessing
df_new = data_prepocessing(df)
df_new.head(5)
df_new.to_csv("dataA_tidy.csv")

# data exploration
data_exploration(df_new)

# ======================================================================================================================
# Task A
labels = df["Label"].to_list()
contents = df["tidy"].to_list()

# td-idf
tf_matrix = tf_idf(contents)

# split data
X_train, X_test, y_train, y_test = train_test_split(tf_matrix, labels, test_size=0.4, random_state=1)

# logistic regression
logistic_reg(X_train, X_test, x_val, y_val, y_train, y_test)

# # random forest
random_forest(X_train, X_test, x_val, y_val, y_train, y_test)

# # svm
svm_training(X_train, X_test, x_val, y_val, y_train, y_test)

# LSTM
# build model
tokenizer, vocab_size = word_to_vector(df)
x_train, x_test, y_train, y_test = data_prepare(tokenizer)
embedding_layer = embedding_layer(vocab_size)
model = build_lstm_model(embedding_layer)
# train lstm model
train_model(model, x_train, x_test, y_train, y_test)
# final test accuracy
score = model.evaluate(x_test, y_test, batch_size=800)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])

# ======================================================================================================================
# Task B
# KNN
topicss = list(set(df["Topic"].to_list()))
print(len(topicss))

arg_val_acc = 0
arg_test_acc = 0
arg_train_acc = 0
for t in topicss:
  df_topic = df[df["Topic"]==t]
  # print(df_topic)
  val_acc, test_val, train_acc = train_model(df_topic)
  arg_val_acc += val_acc
  arg_test_acc += test_val
  arg_train_acc += train_acc
arg_val_acc = arg_val_acc/60
arg_test_acc = arg_test_acc/60
arg_train_acc = arg_train_acc/60
print("arg_val_acc: "+ str(arg_val_acc))
print("arg_test_acc: "+ str(arg_test_acc))
print("arg_train_acc: "+ str(arg_train_acc))

# LSTM
# build model
tokenizer, vocab_size, w2v_model = word_to_vector(df)
x_train, x_test, y_train, y_test = data_prepare(tokenizer)
embedding_layer = embedding_layer(vocab_size, w2v_model)
model = build_lstm_model(embedding_layer)
# train lstm model
train_model(model, x_train, x_test, y_train, y_test)
# final test accuracy
score = model.evaluate(x_test, y_test, batch_size=128)
print()
print("ACCURACY:",score[1])
print("LOSS:",score[0])


