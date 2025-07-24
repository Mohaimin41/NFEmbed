from multiprocessing import Pool
from sklearn.model_selection import train_test_split,StratifiedKFold,GridSearchCV,StratifiedShuffleSplit
import pandas as pd
import numpy as np
import random
from sklearn import svm
import os
import sys
from scipy import stats
import threading
import time
from random import randint, sample
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score, f1_score, matthews_corrcoef, roc_auc_score,roc_curve,auc,balanced_accuracy_score,classification_report
from sklearn.svm import SVC as SVC_gpu
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
import pickle
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef, roc_auc_score,
    average_precision_score
)


random.seed(0)

df=pd.read_csv('dataset.csv', encoding='ISO-8859-1')
#Normalize the features extracted from the ProtT5 model
scaler = MinMaxScaler()
df.iloc[:, 1:1025] = scaler.fit_transform(df.iloc[:, 1:1025])

print(df.shape)

#Validate the performance on the test set
XX = df.iloc[:, 1:1939]
YY = df["label_classification"]
test_index = pd.read_csv('test_index.csv')
indices_test = test_index['test_index']
XX = XX.iloc[indices_test]
YY = YY.iloc[indices_test]
XX = pd.DataFrame(XX)
XX = XX.astype(np.float32)
YY = YY.astype(np.int32)

XX = XX.values

with open('xgb.pkl', 'rb') as f:
    clf2 = pickle.load(f)
s22 = clf2.score(XX, YY)
all_probs = clf2.predict_proba(XX)[:, 1]
all_preds = clf2.predict(XX)

all_probs = np.array(all_probs).reshape(-1)
all_preds = np.array(all_preds).reshape(-1)
all_labels = np.array(YY).reshape(-1)

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)  # Sensitivity
f1 = f1_score(all_labels, all_preds)
mcc = matthews_corrcoef(all_labels, all_preds)
auc = roc_auc_score(all_labels, all_probs)
bal_acc = balanced_accuracy_score(all_labels, all_preds)

# Compute specificity
tn, fp, fn, tp = confusion_matrix(all_labels, all_preds).ravel()
specificity = tn / (tn + fp)

# print all
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Balanced Accuracy: {bal_acc:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"F1: {f1:.4f}")
print(f"MCC: {mcc:.4f}")
print(f"AUC: {auc:.4f}")