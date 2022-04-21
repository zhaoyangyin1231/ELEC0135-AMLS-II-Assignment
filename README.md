# ELEC0135-AMLS-II-Assignment

## Assignment overview
The assignment is to solve the Twitter sentiment analysis tasks. For both task A and task B, classifiers are designed including Random forest, Logistci regression, SVM, KNN, LSTM and Bi-LSTM. Tests are designed to do hyper-parameters selection. The results prove that the models achieve considerable accuracy in both tasks.
## Requirements
-	Python == 3.8
-	scikit-learn == 1.0.1
-	numpy == 1.21.2
-	pandas == 13.4
-	gensim == 4.0.1
-	Keras == 2.3.1
## Datasets
-	training data: 
-	-- task A English: 15552 Twitter texts organized in 3 classes. 
-	-- task A Arabic: 3355 Twitter texts organized in 3 classes. 
-	-- task B English: 4345 Twitter texts organized in 2 classes, 60 topics. 
-	-- task B Arabic: 1656 Twitter texts organized in 2 classes, 34 topics.
-	validation data and test data: 
-	-- split training data into 3:1:1

## Code organization
The code is separated in three files, for data preprocessing, task A and task B. 
Both files include the functions need for data pre-processing and/or model training. To execute the code, for jupyter notebook file, click Run button to run the whole program or click run icon of each line to run the single code block. For Python file, run main.py
## Results
- English dataset:
|       | Model | Training Accuracy | Validation Accuracy | Testing  Accuracy |
| :----: | :----: | :----: | :----: | :----: |
| TaskA | Logistic Regression | 83.0% | 60.9% | 60.8% |
| TaskA | Random Forest | 88.0% | 58.6% | 56.6% |
| TaskA | SVM | 91.3% | 61.2% | 61.0% |
| TaskA | Bi-LSTM | 89.2% | 60.5% | 61.3% |
| TaskB | KNN | 90.8% | 82.3% | 80.4% |
| TaskB | LSTM | 99.8% | 82.3% | 82.1% |
- Arabic dataset:
|       | Model | Training Accuracy | Validation Accuracy | Testing  Accuracy |
| :----: | :----: | :----: | :----: | :----: |
| TaskA | Logistic Regression | 85.0% | 58.4% | 56.3% |
| TaskA | Random Forest | 89.2% | 52.4% | 53.7% |
| TaskA | SVM | 872.3% | 58.1% | 54.3% |
| TaskA | Bi-LSTM | 77.9% | 52.4% | 54.8% |
| TaskB | KNN | 88.2% | 79.9 | 76.9% |
| TaskB | LSTM | 74.8% | 63.6% | 64.3% |
