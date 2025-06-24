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

train_indices = pd.read_csv(f"../../all_features/train_index.csv").values.reshape(-1)
X_train = X[train_indices]
y_train = y[train_indices]

print(X_train.shape, y_train.shape)

X_base, X_meta, y_base, y_meta = train_test_split(X, y, test_size=0.4, random_state=42)

base_model = KNeighborsClassifier(n_neighbors=5, algorithm='auto', leaf_size=30, weights='uniform')
base_model.fit(X_base, y_base)
with open('base_model.pkl', 'wb') as f:
    pickle.dump(base_model, f)

y_meta_proba = base_model.predict_proba(X_meta)[:, 1]

X_meta = np.concatenate((X_meta, y_meta_proba.reshape(-1, 1)), axis=1)
print(X_meta.shape)

meta_model = RandomForestClassifier(random_state=1, n_estimators=200, max_depth=None, min_samples_split=5, max_features='log2', min_samples_leaf=1, bootstrap=True)
meta_model.fit(X_meta, y_meta)

with open('meta_model.pkl', 'wb') as f:
    pickle.dump(meta_model, f)
