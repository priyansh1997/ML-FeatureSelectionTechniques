# -*- coding: utf-8 -*-
"""
Created on Sun May 31 11:51:10 2020

@author: PRIYANSH Feature selection using LDA and PCA
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

data = pd.read_csv(open("E:/ML/feature_selection _methods/santander-train.csv",'rb'),nrows = 20000)


X = data.iloc[:,:-1]

y = data.iloc[:,-1]

const_filter = VarianceThreshold(threshold = 0.01) #keep all features with non-zero variance

const_filter.fit(X)

X_filter = const_filter.transform(X)

X_filter_T = X_filter.T

X_filter_T = pd.DataFrame(X_filter_T)

X_filter_T.duplicated().sum()

duplicated_features = X_filter_T.duplicated()

features_to_keep = [not index for index in duplicated_features]

X_filter_T_uniq = X_filter_T[features_to_keep].T

scaler = StandardScaler().fit(X_filter_T_uniq)

X_filter_t_uniq_scld = scaler.transform(X_filter_T_uniq)

X_filter_T_uniq_scld = pd.DataFrame(X_filter_t_uniq_scld)

corrmat = X_filter_T_uniq_scld.corr()

def get_correlation (data, threshold):
    corr_col = set()
    corrmat = data.corr()

    for i in range(len(corrmat.columns)):
        for j in range(i):
             if abs(corrmat.iloc[i, j]) > threshold:  #here 0.85 is the threshold correlated value above which the features will not be taken
                colname = corrmat.columns[i]
                corr_col.add(colname)
    return corr_col

corr_features = get_correlation(X_filter_T_uniq_scld, 0.70)       

print('no of correlated features: ', len(set(corr_features)))

X_uncorr = X_filter_T_uniq_scld.drop(labels = corr_features, axis = 'columns')

###################feature dimension reduction by LDA###################
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

#lda tries to remove the seprability in classes rather than features
lda = LDA(n_components = 1)

X_lda = lda.fit_transform(X_uncorr, y)


X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size = 0.2)

def rfc(X_train, X_test, y_train, y_test):
    clf = RandomForestClassifier(n_estimators= 100, n_jobs=-1)
    
    clf.fit(X_train,y_train)
    
    y_pred = clf.predict(X_test)
    
    print("accuracy is ",accuracy_score(y_test,y_pred))

rfc(X_train, X_test, y_train, y_test)

####################feature reduction by PCA#################
from sklearn.decomposition import PCA

#it works on features and reduce it to 2 we can select n_components till (n-1) like in X_uncorr we have 82 features so we can use 81
#with the increase in component the accuracy will increase
pca = PCA(n_components=3)
pca.fit(X_uncorr)

X_pca = pca.transform(X_uncorr)

X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size = 0.2)

rfc(X_train_pca, X_test_pca, y_train, y_test)

for components in range(1,79):
    pca = PCA(n_components = components)
    pca.fit(X_uncorr)
    X_pca = pca.transform(X_uncorr)
    print('selected components: ', components)
    X_train_pca1, X_test_pca1, y_train, y_test = train_test_split(X_pca, y, test_size = 0.2)
    rfc(X_train_pca, X_test_pca, y_train, y_test)
    print()





























