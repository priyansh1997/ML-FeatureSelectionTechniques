# -*- coding: utf-8 -*-
"""
Created on Sat May 23 09:44:06 2020

@author: PRIYANSH univariate Anova test and T-test
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import f_classif, f_regression, SelectPercentile, SelectKBest

data = pd.read_csv(open("E:/ML/feature_selection _methods/santander-train.csv",'rb'),nrows = 20000)

X = data.iloc[:,:-1]

y = data.iloc[:,-1]

c_f = VarianceThreshold(threshold = 0.01)

X_c_f = c_f.fit_transform(X)

X_c_f_T = X_c_f.T

X_c_f_T = pd.DataFrame(X_c_f_T)

a = X_c_f_T.duplicated().sum()

duplicated_features = X_c_f_T.duplicated()

features_to_keep = [not index for index in duplicated_features]

X_c_f_u = X_c_f_T[features_to_keep].T

X_train, X_test, y_train, y_test = train_test_split(X_c_f_u, y, test_size = 0.2)

###################################now doing ftest#####################

sel = f_classif(X_train, y_train)

p_values = pd.Series(sel[1])

p_values.index = X_c_f_u.columns

p_values.sort_values(ascending = True, inplace = True)

p_values.plot.bar(figsize = (16,5))

p_values = p_values[p_values < 0.05]
#theory says that those p_values are imp whose values are less than 0.05
p_values.index

X_train_p = X_train[p_values.index]

X_test_p = X_test[p_values.index]

def Rfc(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    
Rfc(X_train_p, X_test_p, y_train, y_test)

Rfc(X_train, X_test, y_train, y_test)


# =============================================================================
# f_tes and anova test are same
# =============================================================================




























































