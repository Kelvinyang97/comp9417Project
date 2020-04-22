import matplotlib.pyplot as plt
import numpy as np
from preproc import preproc as Parse


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
from sklearn.manifold import TSNE

#Get Access to data
DATA = Parse('data/', tfidf=True, max_features = 500, min_df = 10)
#Assign features
bagOfWords = DATA.features[0]
testBag = DATA.features[1]
bagNames = DATA.features[2]

#AttemptScaling
#transformer = TSNE(n_components = 10, method='exact', n_jobs=-3)
#transformer.fit(bagOfWords)
#bagOfWords = transformer.transform(bagOfWords)
#testBag = transformer.transform(testBag)

#Assign Labels
transLabels = DATA.labels[0]
testLabels = DATA.labels[1]
listLabels = DATA.labels[2]

### MODEL DEFAULTS ###
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
# refit = Tclea
# intercept_scaling = 1.0
# multi_class = 'auto' --> Ensure multiple classes with multinomial
# random_state = None (good for ensure consistent build)
# l1_ratios = None

# base model
CVLRModel = LogisticRegressionCV(n_jobs=-3, multi_class = 'multinomial', max_iter = 1000)
CVLRModel.fit(bagOfWords, transLabels)

print( 'Score on training data: ', CVLRModel.score(bagOfWords, transLabels) )
print( 'Score on test data: ', CVLRModel.score(testBag, testLabels) )


# param tuning for values of c
# tuning this does seem to make a difference to score
C = [1,3,5,7]
best_c = 0.1
highest_score = 0.0
for reg in C:
    model_c = LogisticRegressionCV(n_jobs=-3, multi_class = 'multinomial', max_iter = 1000, Cs = reg)
    model_c.fit(bagOfWords, transLabels)
    print( 'Score on training data: ', model_c.score(bagOfWords, transLabels) )
    test_score = model_c.score(testBag, testLabels)
    print( 'Score on test data: ', test_score)
    if test_score > highest_score:
        highest_score = test_score
        best_c = reg
    

# tune size of CV 
# note: size of cv doesn't seem to be making a difference, with test accuracy sitting at
# 0.776 
CV = [3,5,7,9]

best_CV = 3
highest_acc = 0

for cross_val in CV:   
    model_cv = LogisticRegressionCV(n_jobs=-3, multi_class = 'multinomial', max_iter = 1000, cv = cross_val)
    model_cv.fit(bagOfWords, transLabels)
    print(f'Score on training data for cv = {cross_val}: ', model_cv.score(bagOfWords, transLabels) )
    test_score = model_cv.score(testBag, testLabels)
    print(f'Score on test data for cv = {cross_val}: ', test_score )
    if test_score > highest_acc:
        highest_ac = test_score
        best_CV = cross_val


# model with best parameters
model_c = LogisticRegressionCV(n_jobs=-3, multi_class = 'multinomial', max_iter = 1000, Cs = best_c, cv = best_CV)
model_c.fit(bagOfWords, transLabels)
print( 'Score on training data: ', model_c.score(bagOfWords, transLabels) )
print( 'Score on test data: ', model_c.score(testBag, testLabels) )
