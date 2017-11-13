import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import roc_auc_score

bank = pd.read_csv('data/Bank_data_to_class.csv', sep=',')

# Write your solution here :
y = bank['y']
X = bank.drop(['y'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=9)
# Write your solution here :
model = RandomForestClassifier(random_state=9,class_weight = 'balanced')

def pipeline(X_train, X_test, y_train, y_test,model):
    param_grid = {"max_depth": [2, 3, 5, 6, 8, 10, 15, 20, 30],
              "max_leaf_nodes": [2, 5, 10, 15, 20],
              "max_features": [8,10,12,14]}
    grid = GridSearchCV(estimator=model,param_grid=param_grid)
    label = LabelEncoder()
    y_train=label.fit_transform(y_train)
    for column in X_train.columns:
        if X_train[column].dtype == type(object):
            label = LabelEncoder()
            X_train[column] = label.fit_transform(X_train[column])
    y_test=label.fit_transform(y_test)
    for column in X_test.columns:
        if X_test[column].dtype == type(object):
            label = LabelEncoder()
            X_test[column] = label.fit_transform(X_test[column])
    grid.fit(X_train, y_train)
    auc= roc_auc_score(y_test, grid.predict(X_test))
    return grid.fit(X_train, y_train),auc
