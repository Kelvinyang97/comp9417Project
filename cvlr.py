import matplotlib.pyplot as plt
import numpy as np
from preproc import preproc as Parse


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV

#Get Access to data
DATA = Parse('data/', tfidf=True, max_df = 0.3)

#Assign features
bagOfWords = DATA.features[0]
testBag = DATA.features[1]
bagNames = DATA.features[2]

#Assign Labels
transLabels = DATA.labels[0]
testLabels = DATA.labels[1]
listLabels = DATA.labels[2]

### MODEL DEFAULS ###
# Cs = 10
# fit_intercept = T
# cv = 5-fold
# dual = F
# penalty = l2
# scoring = accuracy
# solver = lbfgs <- maybe test these
# tol = 1e-4
# max_iter = 100
# class_weight = none
# n_jobs = None --> This is the number of cores/processors being used to solve it
# -1 means ALL processors, -2 means all but 1, -3 all but 2...
# refit = T
# intercept_scaling = 1.0
# multi_class = 'auto' --> Ensure multiple classes with multinomial
# random_state = None (good for ensure consistent build)
# l1_ratios = None

CVLRModel = LogisticRegressionCV(n_jobs=-3, multi_class = 'multinomial')
CVLRModel.fit(bagOfWords, transLabels)

print( 'Score on training data: ', CVLRModel.score(bagOfWords, transLabels) )
print( 'Score on test data: ', CVLRModel.score(testBag, testLabels) )
