# -*- coding: utf-8 -*-
"""
Created on Sat May 23 11:43:33 2020

@author: PRIYANSH fischer score and chi square
"""
# =============================================================================
# works only on categorical features
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif, f_regression, SelectPercentile, SelectKBest

# =============================================================================
# we remove those features which are irrelevant for the classification or not dependent to the target
# =============================================================================

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

#first array of f_score have f values and second array have p values
f_score = chi2(X_train, y_train)

p_values = pd.Series(f_score[1], index = X_train.columns)

p_values.sort_values(ascending = True, inplace = True)

p_values.plot.bar()

X_train_2 = X_train[['who','sex','pclass','alone']]

X_test_2 = X_test[['who','sex','pclass','alone']]

def Rfc(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators = 100, n_jobs = -1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    
Rfc(X_train_2, X_test_2, y_train, y_test)


#here when we are taking first four p value columns we are getting approx 82% of accuracy and after adding fifth p value we are getting 81% of accuracy so we will take first four p values












































































