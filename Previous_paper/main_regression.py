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
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error


random.seed(0)

df=pd.read_csv('dataset.csv', encoding='ISO-8859-1')
#Normalize the features extracted from the ProtT5 model
scaler = MinMaxScaler()
df.iloc[:, 1:1025] = scaler.fit_transform(df.iloc[:, 1:1025])

print(df.shape)

#Validate the performance on the test set
XX = df.iloc[:, 1:1939]
YY = df["label_regression"]
test_index = pd.read_csv('test_index.csv')
indices_test = test_index['test_index']
XX = XX.iloc[indices_test]
YY = YY.iloc[indices_test]
XX = pd.DataFrame(XX)
XX = XX.astype(np.float32)
YY = YY.astype(np.float32)

XX = XX.values

with open('svr_1.pkl', 'rb') as f:
    clf1 = pickle.load(f)

predicted = clf1.predict(XX[:, :1024])

XX = np.concatenate((XX, predicted.reshape(-1, 1)), axis=1)

with open('svr_2.pkl', 'rb') as f:
    clf2 = pickle.load(f)

s22 = clf2.predict(XX[:, 1024:])

# print scores upto 4 decimal places
r2 = r2_score(YY, s22)
print(f"R2: {r2:.4f}")
mae = mean_absolute_error(YY, s22)
print(f"MAE: {mae:.4f}")
mse = mean_squared_error(YY, s22)
print(f"MSE: {mse:.4f}")
rmse = np.sqrt(mse)
print(f"RMSE: {rmse:.4f}")
