# -*- coding: utf-8 -*-
"""
Created on Sun May 31 09:12:41 2020

@author: PRIYANSH Recursive feature selection using tree based and gradient boosting
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectFromModel

# =============================================================================
# it eliminates the feature recursively means it makes the model and removes the least
# performing feature again it makes the model and again removes the least performing model
# this process goes on to get the max accuracy
# =============================================================================

from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

data.keys()

X = pd.DataFrame(data.data, columns = data.feature_names)

y = data.target

X.isnull().sum()


##################Feature selection by feature importance Random Forest Classifier###############

sel = SelectFromModel(RandomForestClassifier(n_estimators = 100, n_jobs = -1))

sel.fit(X, y)

sel.get_support()

feature_selected = X.columns[sel.get_support()]

len(feature_selected)

np.mean(sel.estimator_.feature_importances_)

#it will show the feature importance
sel.estimator_.feature_importances_

X_rfc = sel.transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_rfc, y, test_size = 0.2, random_state = 0)

def Rfc(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))

Rfc(X_train, X_test, y_train, y_test)

########Recursive feature elimination###########
from sklearn.feature_selection import RFE

sel_rfe = RFE(RandomForestClassifier(n_estimators = 100, n_jobs = -1))

sel_rfe.fit(X, y)

sel_rfe.get_support()

feature_selected_rfe = X.columns[sel_rfe.get_support()]

X_rfe = sel_rfe.transform(X)

X_train_rfe, X_test_rfe, y_train, y_test = train_test_split(X_rfe, y, test_size = 0.2, random_state = 0)

Rfc(X_train_rfe, X_test_rfe, y_train, y_test)

###########feature selection by Gradient Boosting tree importance############
from sklearn.ensemble import GradientBoostingClassifier

sel_gbc = RFE(GradientBoostingClassifier(n_estimators = 100))

sel_gbc.fit(X, y)

sel_gbc.get_support()

feature_selected_gbc = X.columns[sel_gbc.get_support()]

X_gbc = sel_gbc.transform(X)

X_train_gbc, X_test_gbc, y_train, y_test = train_test_split(X_gbc, y, test_size = 0.2, random_state = 0)

Rfc(X_train_gbc, X_test_gbc, y_train, y_test)

for index in range(1, 31):
    sel_gbc = RFE(GradientBoostingClassifier(n_estimators = 100), n_features_to_select = index)
    sel_gbc.fit(X, y)
    X_gbc = sel_gbc.transform(X)
    print('seleced feature: ', index)
    Rfc(X_train_gbc, X_test_gbc, y_train, y_test)
    print()


sel_gbc = RFE(GradientBoostingClassifier(n_estimators = 100), n_features_to_select = 6)
sel_gbc.fit(X, y)
X_gbc = sel_gbc.transform(X)
print('selected feature: ', 6)
Rfc(X_train_gbc, X_test_gbc, y_train, y_test)
print()

feature = X.columns[sel.get_support()]



for index in range(1, 31):
    sel_rfe = RFE(RandomForestClassifier(n_estimators = 100), n_features_to_select = index)
    sel_rfe.fit(X, y)
    X_rfe = sel_rfe.transform(X)
    print('seleced feature: ', index)
    Rfc(X_train_rfe, X_test_rfe, y_train, y_test)
    print()














