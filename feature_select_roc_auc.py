# -*- coding: utf-8 -*-
"""
Created on Fri May 22 10:17:15 2020

@author: PRIYANSH feature selection by ROC_AUC for classification
"""

# =============================================================================
# area under the ROC curve shows the matrix which is AUC
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold

data = pd.read_csv(open("E:/ML/feature_selection _methods/santander-train.csv",'rb'),nrows = 20000)

#data.head

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

# =============================================================================
# now calculating roc and auc score
#takes 0 and 1 
#ROC curves should be used when there are roughly equal numbers of observations for each class
#Precision-Recall curves should be used when there is a moderate to large class imbalance
# ROC curve with an imbalanced dataset might be deceptive and lead to incorrect interpretations of the model skill
# =============================================================================
X_test, X_train, y_test, y_train = train_test_split(X_c_f_u, y, test_size = 0.2)

roc_auc = []

for feature in X_c_f_u:
    clf = RandomForestClassifier(n_estimators = 1000)
    clf.fit(X_train[feature].to_frame(), y_train)
    y_pred = clf.predict(X_test[feature].to_frame())
    roc_auc.append(roc_auc_score(y_test, y_pred))
    
roc_values = pd.Series(roc_auc)

roc_values.index = X_train.columns

roc_values.sort_values(ascending = False, inplace = True)    
    
roc_values.plot.bar()

sel = roc_values[roc_values > 0.5]

X_train_roc = X_train[sel.index]

X_test_roc =  X_test[sel.index]

#now using this X_train_roc we will train our model and get the accu
# =============================================================================
# feature selection using RMSE in Regression
# =============================================================================

from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X, y = load_boston(return_X_y = True)

X = pd.DataFrame(X)  

X.head
  
#stratify of train_test_split does not work for  regression problem

X_train, X_test, y_train, y_test = train_test_split(X, y , test_size = 0.2)

mse = []

for feature in X_train.columns:
    clf = LinearRegression()
    clf.fit(X_train[feature].to_frame(), y_train)
    y_pred = clf.predict(X_test[feature].to_frame())
    mse.append(mean_squared_error(y_pred, y_test))

mse = pd.Series(mse, index = X_train.columns)

mse.sort_values(ascending = False, inplace = True)

#higher the mse means more error less mse means they are imp features

mse.plot.bar()

X_train_2 = X_train[[12,5]]

X_test_2 = X_test[[12,5]]


model = LinearRegression()

model.fit(X_train_2, y_train)

y_pred = model.predict(X_test_2)

rscore = r2_score(y_test, y_pred)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

standard_deviation = np.std(y)

# =============================================================================
# rmse < standard_deviation so it is good selection
# =============================================================================

#now doing on original dataset we will get the difference in time of output
#univariate method does  not perform better than wrapper method and embedded method
