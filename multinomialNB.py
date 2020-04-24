#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# In[11]:


from preproc import preproc
file = preproc(filepath = '', trainname = 'data/training.csv', testname = 'data/test.csv', tfidf = False, minmax = False, min_range = 1,
                 max_range = 1, max_df = 1.0, min_df =1, max_features = None)
#x_train:bag_of_words
x_train = file.features[0]
#y_train:transformed_labels
y_train = file.labels[0]
x_test = file.features[1]
y_test = file.labels[1]


# In[24]:

#This works a bit better
"""mnbc = MultinomialNB()
mnbc.fit(x_train, y_train)
mnbc_pred = mnbc.predict(x_test)
# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(y_train, mnbc.predict(x_train)))
# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(y_test, mnbc_pred))"""


# In[3]:


from sklearn.model_selection import GridSearchCV
#Tune Hyperparameters
#test grid_search
#grid search cross validation
def getAP(a,d,n): 
    res = []
    curr_term = a
    res.append(a)
    for i in range(1,n):
        curr_term =curr_term + d 
        res.append(curr_term)
    return res


# In[4]:


mnbc = MultinomialNB()
alpha_range = getAP(1.2, 0.01, 50)
param_grid = {'alpha': alpha_range}
grid_search = GridSearchCV( estimator=mnbc,
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=3)
grid_search.fit(x_train, y_train)
print("The mean accuracy of a model with these hyperparameters is:")
print(grid_search.best_score_)


# In[5]:


from sklearn.model_selection import RandomizedSearchCV
#Tune Hyperparameters
#test Randomized Search
#Randomized Search cross validation


# In[6]:


mnbc = MultinomialNB()
alpha_range = getAP(0.1, 0.01, 500)
random_grid = {'alpha': alpha_range}
random_search = RandomizedSearchCV(estimator=mnbc,
                                   param_distributions=random_grid,
                                   n_iter=50,
                                   scoring='accuracy',
                                   cv=3)
random_search.fit(x_train, y_train)
print("The mean accuracy of a model with these hyperparameters is:")
print(random_search.best_score_)

# In[17]:


best_mnbc = grid_search.best_estimator_
best_mnbc.fit(x_train, y_train)
mnbc_pred = best_mnbc.predict(x_test)


# In[20]:


# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(y_train, best_mnbc.predict(x_train)))


# In[19]:


# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(y_test, mnbc_pred))


from sklearn.model_selection import KFold
from sklearn.ensemble import BaggingClassifier
seed = 8
kfold = KFold(n_splits=10, random_state=seed)
bagged_base = grid_search.best_estimator_
num_estimator = 5
model = BaggingClassifier(base_estimator=bagged_base, n_estimators=num_estimator, random_state=seed)
model.fit(x_train, y_train)
train_results = model.predict(x_train)
print("The training accuracy for bagging is: ")
print(accuracy_score(y_train, train_results))
test_results = model.predict(x_test)
print("The training accuracy for bagging is: ")
print(accuracy_score(y_test, test_results))


# In[21]:


print(classification_report(y_test,mnbc_pred))

