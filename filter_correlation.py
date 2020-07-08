# -*- coding: utf-8 -*-
"""
Created on Sun May 10 09:42:43 2020

@author: PRIYANSH FEATURE SELECTION METHODS
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
 

from sklearn.model_selection import train_test_split

file = pd.read_csv(open('E:/ML/santander-train.csv','rb'))

X = file.iloc[:,0:370]

y = file.iloc[:,370]


"""
types of methods for feature selection:- 
1. filter method
    it doesnt involve any machine learning model while selecting a feature
    these are quite fast methods
    ideal for removing constant and quasi constant features(quasi means not really)
    this is also of 2 types univariate and multivariate
    
    firstly we are removing constant features
"""
from sklearn.feature_selection import VarianceThreshold 

const_filter = VarianceThreshold(threshold = 0) #keep all features with non-zero variance

const_filter.fit(X)

x = const_filter.get_support().sum() # we can use the get_support() method of the filter that we created

c = [not temp for temp in const_filter.get_support()] #return true for all the columns which are constant 

X.columns[c] #for getting the names of unwanted columns

#######################################updating training and testing dataset

X_filtered = const_filter.transform(X)

######################################################################

"""
for removing quasi constant features 
we will take threshold to be 0.1 or lesser than it to remove those
values which are likely to be constant
"""


q_const_filter = VarianceThreshold(threshold = 0.01)

q_const_filter.fit(X_filtered)

X_q_filtered = q_const_filter.transform(X_filtered)

"""
###############################now we will remove duplicate features


X_T = X_q_filtered.T #transposed

X_T = pd.DataFrame(X_T)

a = X_T.duplicated().sum()

duplicated_features = X_T.duplicated() #duplicated rows will be true

features_to_keep_a = [not index for index in duplicated_features] 

X_notDuplicated = X_T[features_to_keep_a].T


#############################################################################
"""



"""
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_q_filtered, y, test_size = 0.2, stratify = y)

clf = RandomForestClassifier(n_estimators= 100,random_state= 1, n_jobs=-1)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("accuracy is ",accuracy_score(y_test,y_pred))
"""



"""
correlated feature removal

2 or more than 2 features are correlated with each other if they are close in there linear space
our main goal is to remove the correlated features to reduce the dimensionality of our model
colinearity among the features is not required

"""
"""
#for heat mapping......................
import seaborn as sns #Seaborn is a Python data visualization library based on matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
plt.figure(figsize = (12,8))#12 is width and 8 is hieght

-------------------------------------------------------------------------------------------------
numint or str, optional, default: None
If not provided, a new figure will be created, and the figure number will be incremented. The figure objects holds this number in a number attribute. If num is provided, and a figure with this id already exists, make it active, and returns a reference to it. If this figure does not exists, create it and returns it. If num is a string, the window title will be set to this figure's num.

figsize(float, float), optional, default: None
width, height in inches. If not provided, defaults to rcParams["figure.figsize"] (default: [6.4, 4.8]) = [6.4, 4.8].

dpiinteger, optional, default: None
resolution of the figure. If not provided, defaults to rcParams["figure.dpi"] (default: 100.0) = 100.

facecolorcolor
the background color. If not provided, defaults to rcParams["figure.facecolor"] (default: 'white') = 'w'.

edgecolorcolor
the border color. If not provided, defaults to rcParams["figure.edgecolor"] (default: 'white') = 'w'.

frameonbool, optional, default: True
If False, suppress drawing the figure frame.

FigureClasssubclass of Figure
Optionally use a custom Figure instance.

clearbool, optional, default: False
If True and the figure already exists, then it is cleared.
-----------------------------------------------------------------------------------


sns.heatmap(correlation_matrix, 
            xticklabels=correlation_matrix.columns,
            yticklabels=correlation_matrix.columns,
            cmap='RdBu_r',
            annot=True,
            linewidth=0.5)


#now we will filter both the +ve and -ve correlated features by absoluting all the values
"""


X_q_filtered = pd.DataFrame(X_q_filtered)
#we will add .values to convert dataframe into matrix
correlation_matrix = X_q_filtered.corr().values


corr_col = set()
for i in range(273):
    for j in range(i):
         if abs(correlation_matrix.iloc[i,j]) > 0.85:  #here 0.85 is the threshold correlated value above which the features will not be taken
            colname = correlation_matrix.columns[i]
            corr_col.add(colname)
            
X_uncorr = X_q_filtered.drop(labels = corr_col, axis = 'columns')

"""
.drop is used removing unwanted columns

labels single label or list-like
Index or column labels to drop.

axis{0 or ‘index’, 1 or ‘columns’}, default 0
Whether to drop labels from the index (0 or ‘index’) or columns (1 or ‘columns’).

indexsingle label or list-like
Alternative to specifying axis (labels, axis=0 is equivalent to index=labels).

New in version 0.21.0.

columnssingle label or list-like
Alternative to specifying axis (labels, axis=1 is equivalent to columns=labels).

New in version 0.21.0.

levelint or level name, optional
For MultiIndex, level from which the labels will be removed.

inplacebool, default False
If True, do operation inplace and return None.

errors{‘ignore’, ‘raise’}, default ‘raise’
If ‘ignore’, suppress error and only existing labels are dropped.    

##################it is not recommended because there may be some important features which must be used for knowing the predictions

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X_uncorr, y, test_size = 0.2, stratify = y)

clf = RandomForestClassifier(n_estimators= 100,random_state= 1, n_jobs=-1)

clf.fit(X_train,y_train)

y_pred = clf.predict(X_test)

print("accuracy is ",accuracy_score(y_test,y_pred))
"""

correlation_matrix_stack = correlation_matrix.abs().stack()

cm_sort = correlation_matrix_stack.sort_values(ascending=False)

cm_sort = cm_sort[cm_sort > 0.85]

cm_sort = cm_sort[cm_sort < 1]
# =============================================================================
# here we are getting the values with there column index no so we are using .sort_values() method its parameters are:
# by: Single/List of column names to sort Data Frame by.
# axis: 0 or ‘index’ for rows and 1 or ‘columns’ for Column.
# ascending: Boolean value which sorts Data frame in ascending order if True.
# inplace: Boolean value. Makes the changes in passed data frame itself if True.
# kind: String which can have three inputs(‘quicksort’, ‘mergesort’ or ‘heapsort’) of algorithm used to sort data frame.
# na_position: Takes two string input ‘last’ or ‘first’ to set position of Null values. Default is ‘last’.
# =============================================================================

cm_sort = pd.DataFrame(cm_sort).reset_index()
# =============================================================================
# Syntax:
# DataFrame.reset_index(level=None, drop=False, inplace=False, col_level=0, col_fill=”)
# Parameters:
# level: int, string or a list to select and remove passed column from index.
# drop: Boolean value, Adds the replaced index column to the data if False.
# inplace: Boolean value, make changes in the original data frame itself if True.
# col_level: Select in which column level to insert the labels.
# col_fill: Object, to determine how the other levels are named.
# 
# Return type: DataFrame
# =============================================================================

cm_sort.columns = ['feature_1','feature_2','label'] #renaming the columns


grouped_features = []
correlated_feature_list = []

lis = cm_sort.feature_1
for i in lis.unique():
    if i not in grouped_features:
        correlated_block = cm_sort[lis == i]
        #grouped_features = grouped_features + list(correlated_block.feature_2.unique()) + [i]
        correlated_feature_list.append(correlated_block)

for i in correlated_feature_list:
    print(i)

###############################important features#############################

important_features = []
from sklearn.ensemble import RandomForestClassifier
X_train, X_test, y_train, y_test = train_test_split(X_uncorr, y, test_size = 0.2, stratify = y)

for i in correlated_feature_list:
    feature = list(i.feature_1.unique()) + list(i.feature_2.unique())# here i is used so that we can get the list with separate groups
    clf = RandomForestClassifier(n_estimators= 100,random_state= 1, n_jobs=-1)
    clf.fit(X_train, y_train)
# =============================================================================
#     Concatenate pandas objects along a particular axis with optional set logic along the other axes.
# 
# Can also add a layer of hierarchical indexing on the concatenation axis, which may be useful if the labels are the same (or overlapping) on the passed axis number.
# 
# Parameters
# objsa sequence or mapping of Series or DataFrame objects
# If a dict is passed, the sorted keys will be used as the keys argument, unless it is passed, in which case the values will be selected (see below). Any None objects will be dropped silently unless they are all None in which case a ValueError will be raised.
# 
# axis{0/’index’, 1/’columns’}, default 0
# The axis to concatenate along.
# 
# join{‘inner’, ‘outer’}, default ‘outer’
# How to handle indexes on other axis (or axes).
# 
# ignore_indexbool, default False
# If True, do not use the index values along the concatenation axis. The resulting axis will be labeled 0, …, n - 1. This is useful if you are concatenating objects where the concatenation axis does not have meaningful indexing information. Note the index values on the other axes are still respected in the join.
# 
# keyssequence, default None
# If multiple levels passed, should contain tuples. Construct hierarchical index using the passed keys as the outermost level.
# 
# levelslist of sequences, default None
# Specific levels (unique values) to use for constructing a MultiIndex. Otherwise they will be inferred from the keys.
# 
# nameslist, default None
# Names for the levels in the resulting hierarchical index.
# 
# verify_integritybool, default False
# Check whether the new concatenated axis contains duplicates. This can be very expensive relative to the actual data concatenation.
# 
# sortbool, default False
# Sort non-concatenation axis if it is not already aligned when join is ‘outer’. This has no effect when join='inner', which already preserves the order of the non-concatenation axis.
# 
# New in version 0.23.0.
# 
# Changed in version 1.0.0: Changed to not sort by default.
# 
# copybool, default True
# If False, do not copy data unnecessarily.
# 
# Returns
# object, type of objs
# =============================================================================
    importance = pd.concat([pd.Series(feature), pd.Series(clf.feature_importances_)], axis = 1)
    importance.columns = ['feature', 'importance']
    importance.sort_values(by = 'importance', ascending = False, inplace = True)
    feat = importance.iloc[0]
    important_features.append(feat)
    

important_features=pd.DataFrame(important_features)
important_features.reset_index(inplace = True, drop = True)


feature_to_consider = set(important_features['feature'])
feature_to_discard = set(corr_feature) - set(feature_to_consider)


X_uncorr




















































































 