from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
import numpy as np
import random
import csv
from sklearn.metrics import roc_curve
import lightgbm as lgb
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.model_selection import train_test_split
import pickle

random.seed(0)
np.random.seed(0)


def find_metrics(y_test, y_predict, y_proba):

    tn, fp, fn, tp = confusion_matrix(y_test, y_predict).ravel()  # y_true, y_pred

    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    acc = accuracy_score(y_test, y_predict)
    prec = tp / (tp + fp)
    f1_score_1 = 2 * prec * sensitivity / (prec + sensitivity)
    mcc = matthews_corrcoef(y_test, y_predict)
    auc = roc_auc_score(y_test, y_proba[:, 1])
    bal_acc = balanced_accuracy_score(y_test, y_predict)

    return sensitivity, specificity, acc, prec, f1_score_1, mcc, auc, bal_acc


Models = ['RF', 'SVM', 'KNN', 'XGB', 'LBM', 'ADA', 'GBC', 'GPC', 'QDA', 'MLP', 'CAT']

feature_groups_single = ['CT', 'RSCU', 'Copy_number', 'Gene_distance']

feature_arrays = []

for feature_group in feature_groups_single:
    print(f"Processing {feature_group}...")
    # Load each feature group as a NumPy array
    X = np.load(f"../../all_features/{feature_group}.npy")
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    feature_arrays.append(X)

X_nifd = np.load(f"../../all_features/nifd_600m.npy")
X_nifh = np.load(f"../../all_features/nifh_600m.npy")
X_nifk = np.load(f"../../all_features/nifk_600m.npy")

X = (X_nifd + X_nifh + X_nifk) / 3

feature_arrays.append(X)
# Concatenate all features on the column axis (axis=1)
X = np.concatenate(feature_arrays, axis=1)

print(X.shape)
y = np.load(f"../../all_features/labels.npy")

test_indices = pd.read_csv(f"../../all_features/test_index.csv").values.reshape(-1)

X_test = X[test_indices]
y_test = y[test_indices]

with open(f"base_model.pkl", 'rb') as f:
    base_model = pickle.load(f)

with open(f"meta_model.pkl", 'rb') as f:
    meta_model = pickle.load(f)

y_test_proba = base_model.predict_proba(X_test)[:, 1]
X_test = np.concatenate((X_test, y_test_proba.reshape(-1, 1)), axis=1)
# y_predict = meta_model.predict(X_test)
y_proba = meta_model.predict_proba(X_test)

y_predict = [1 if prob[1] > 0.55 else 0 for prob in y_proba]

sensitivity, specificity, acc, prec, f1_score_1, mcc, auc, bal_acc = find_metrics(y_test, y_predict, y_proba)

print(f"Sensitivity: {sensitivity}")
print(f"Specificity: {specificity}")
print(f"Accuracy: {acc}")
print(f"Precision: {prec}")
print(f"F1 Score: {f1_score_1}")
print(f"MCC: {mcc}")
print(f"AUC: {auc}")
print(f"Balanced Accuracy: {bal_acc}")

# save metrics to csv upto 3 decimal place
writer = open('./results.csv', 'w')
csvWriter = csv.writer(writer)

csvWriter.writerow(['Model', 'SN', 'SP', 'BACC', 'ACC', 'PREC', 'F1', 'MCC', 'AUC'])
# upto 3 decimal places
csvWriter.writerow(['Meta Model', 
                    f"{sensitivity:.3f}",
                    f"{specificity:.3f}",
                    f"{bal_acc:.3f}",
                    f"{acc:.3f}",
                    f"{prec:.3f}",
                    f"{f1_score_1:.3f}",
                    f"{mcc:.3f}",
                    f"{auc:.3f}"])