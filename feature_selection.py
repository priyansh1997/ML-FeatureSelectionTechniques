# -*- coding: utf-8 -*-
"""
Created on Sun May 17 10:31:03 2020

@author: PRIYANSH
"""

import pandas as pd
import numpy as np

file = pd.read_csv(open('glass.csv','rb'))

X = file.iloc[:,0:8]

y = file.iloc[:,9]

X.columns

len(X.columns)

X.head()

X.shape

#len(X.Na.unique())

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.ensemble import RandomForestClassifier

r_f_c = RandomForestClassifier(n_estimators = 100, random_state = 5)

r_f_c.fit(X_train, y_train)

y_pred = r_f_c.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))