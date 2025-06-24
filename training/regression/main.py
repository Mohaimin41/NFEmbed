from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import pandas as pd
import numpy as np
import random
import csv
from sklearn.metrics import roc_curve
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
import pickle

random.seed(0)
np.random.seed(0)

def find_metrics(y_test, y_pred):
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)

    return r2, mse, rmse, mae


feature_groups_single = ['Gene_distance', 'Euclidean_distance', 'Copy_number', 'Expression']

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
y = np.load(f"../../all_features/labels_regression.npy")

train_indices = pd.read_csv(f"../../all_features/train_index.csv").values.reshape(-1)
X_train = X[train_indices]
y_train = y[train_indices]

print(X_train.shape, y_train.shape)

X_base, X_meta, y_base, y_meta = train_test_split(X, y, test_size=0.4, random_state=42)

base_model_1 = DecisionTreeRegressor(random_state=42, max_depth=5, min_samples_split=10, min_samples_leaf=1, max_features='sqrt', splitter='random')
base_model_2 = XGBRegressor(
            objective='reg:squarederror',  # Required for regression
            n_estimators=200,
            learning_rate=0.01,
            max_depth=5,
            gamma=0,
            random_state=42
        )
base_model_1.fit(X_base, y_base)
base_model_2.fit(X_base, y_base)
with open('base_model_1.pkl', 'wb') as f:
    pickle.dump(base_model_1, f)

with open('base_model_2.pkl', 'wb') as f:
    pickle.dump(base_model_2, f)

y_meta_pred_1 = base_model_1.predict(X_meta).reshape(-1, 1)
y_meta_pred_2 = base_model_2.predict(X_meta).reshape(-1, 1)

X_meta = np.concatenate((X_meta, y_meta_pred_1, y_meta_pred_2), axis=1)
print(X_meta.shape)

meta_model = SVR(kernel='rbf', C=1.0, epsilon=0.01, gamma='scale')
meta_model.fit(X_meta, y_meta)

with open('meta_model.pkl', 'wb') as f:
    pickle.dump(meta_model, f)


