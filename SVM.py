#!/usr/bin/env python
# coding: utf-8

# In[5]:


#SVM
# First create the base model to tune
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from preproc import preproc


# In[6]:


# Parameter election
min_df = 0.04
max_df = 0.3
max_features = 210


# In[7]:


from preproc import preproc
file = preproc(filepath = '', trainname = 'data/training.csv', testname = 'data/test.csv', tfidf = True, minmax = False, min_range = 1,
                 max_range = 1, max_df = max_df, min_df =min_df, max_features = None)
#x_train:bag_of_words
x_train = file.features[0]
#y_train:transformed_labels
y_train = file.labels[0]

x_test = file.features[1]
y_test = file.labels[1]


# In[ ]:

# first find the rough regions of parameters with randomized search
# C
C = [.0001, .001, .01]
# gamma
gamma = [.0001, .001, .01, .1, 1, 10, 100]
# degree
degree = [1, 2, 3, 4, 5]
# kernel
kernel = ['linear', 'rbf', 'poly']
# probability
probability = [True]
# Create the random grid
random_grid = {'C': C,
              'kernel': kernel,
              'gamma': gamma,
              'degree': degree,
              'probability': probability
             }
#create the model
svc = svm.SVC(random_state=8)
# Definition of the random search
random_search = RandomizedSearchCV(estimator=svc, param_distributions=random_grid, n_iter=3, scoring='accuracy', cv=3, verbose=1, random_state=8)

# Fit the random search model
random_search.fit(x_train, y_train)
print("The best hyperparameters from Random Search are:")
print(random_search.best_params_)
print("")
print("The mean accuracy of a model with these hyperparameters is:")
print(random_search.best_score_)



#Tune paramater with grid_search
def getAP(a,d,n): 
    res = []
    curr_term = a
    res.append(a)
    for i in range(1,n):
        curr_term =curr_term + d 
        res.append(curr_term)
    return res

C = getAP(.0005, .0005, 3)
degree = [1, 2, 3]
gamma = getAP(.005, .005, 3)
probability = [True]

param_grid = [
  {'C': C, 'kernel':['rbf'], 'gamma':gamma, 'probability':probability}
]

#create the model
svc = svm.SVC(random_state=8)
#creat the split
cv_sets = ShuffleSplit(test_size = .33, n_splits = 3, random_state = 8)
#apply the model
grid_search = GridSearchCV(estimator=svc, 
                           param_grid=param_grid,
                           scoring='accuracy',
                           cv=cv_sets,
                           verbose=1)
# Fit the grid search to the data
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)


# In[ ]:


#save the model in best_svc
best_svc = svm.SVC(C=0.0005, break_ties=False, cache_size=200,
                                 class_weight=None, coef0=0.0,
                                 decision_function_shape='ovr', degree=3,
                                 gamma=0.005, kernel='rbf', max_iter=-1,
                                 probability=True, random_state=8,
                                 shrinking=True, tol=0.001, verbose=False)
#fit the training data
best_svc.fit(x_train,y_train)
#get the prediction
svc_pred = best_svc.predict(x_test)
# Training accuracy
print("The training accuracy is: ")
print(accuracy_score(y_train, best_svc.predict(x_train)))
# Test accuracy
print("The test accuracy is: ")
print(accuracy_score(y_test, svc_pred))


# In[ ]:


print(classification_report(y_test,svc_pred))


# In[ ]:




