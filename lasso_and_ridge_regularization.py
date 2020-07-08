# -*- coding: utf-8 -*-
"""
Created on Fri May 29 15:11:50 2020

@author: PRIYANSH Linear and logistic regression coeff with lasso(L1) and Ridge(L2) Regularization for feature Selection in ML
"""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.feature_selection import SelectFromModel


titanic = sns.load_dataset('titanic')

#to check the null values in features of dataset
titanic.isnull().sum()

# in axis = 0 means droping from index and axis = 1 means droping from columns 
# inplace If True, do operation inplace and return None.
# labels Index or column labels to drop.
titanic.drop(labels = ['age','deck'],axis = 1, inplace = True)

titanic = titanic.dropna()

data = titanic[['pclass','sex','sibsp','parch','embarked','who','alone']].copy()


sex = {'male':0, 'female':1}

data['sex'] = data['sex'].map(sex)


embarked = {'S':0, 'C':1, 'Q':2}

data['embarked'] = data['embarked'].map(embarked)


who = {'man':0, 'woman':1, 'child':2}

data['who'] = data['who'].map(who)


alone = {True:0, False:1}

data['alone'] = data['alone'].map(alone)

X = data.copy()

y = titanic['survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

################Estimation of coefficients of linear regression###############

sel = SelectFromModel(LinearRegression())

sel.fit(X_train, y_train)

sel.get_support()

#here those values are selected whose coeff values are greater than its mean value
np.abs(sel.estimator_.coef_)
 
np.mean(np.abs(sel.estimator_.coef_))

#now the three most imp features
features = X_train.columns[sel.get_support()]

X_train_reg = sel.transform(X_train)

X_test_reg = sel.transform(X_test)

def Rfc(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    
Rfc(X_train_reg, X_test_reg, y_train, y_test)

Rfc(X_train, X_test, y_train, y_test)


###############Logistic Regression Coefficient with L1 Regularization################
sel_1 = SelectFromModel(LogisticRegression(penalty = 'l1', C = 0.01, solver = 'liblinear'))
sel_1.fit(X_train, y_train)
sel_1.get_support()

sel_1.estimator_.coef_

X_train_l1 = sel_1.transform(X_train)

X_test_l1 = sel_1.transform(X_test)

Rfc(X_train_l1, X_test_l1, y_train, y_test)

###########L2 Regularization###############
sel_2 = SelectFromModel(LogisticRegression(penalty = 'l2', C = 0.05, solver = 'liblinear'))
sel_2.fit(X_train, y_train)
sel_2.get_support()

sel_2.estimator_.coef_

X_train_l2 = sel_2.transform(X_train)

X_test_l2 = sel_2.transform(X_test)

Rfc(X_train_l2, X_test_l2, y_train, y_test)













