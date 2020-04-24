import matplotlib.pyplot as plt
import numpy as np
from preproc import preproc as Parse

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.decomposition import TruncatedSVD



#Get Access to data
for max in range(2000, 11000, 1000):
    DATA = Parse('data/', tfidf=True, min_df = 10, max_features = max)
    #Assign features
    bagOfWords = DATA.features[0]
    testBag = DATA.features[1]
    bagNames = DATA.features[2]

    #Assign Labels
    transLabels = DATA.labels[0]
    testLabels = DATA.labels[1]
    listLabels = DATA.labels[2]

    #Prepare for TSVD iteration
    tsvd = TruncatedSVD(n_components=bagOfWords.shape[1]-1)
    tsvdFitted = tsvd.fit(bagOfWords)
    #Percentage of variance explained by each of the selected components
    #Essentially sums up to the amount of explanatory power each component adds to the bagofwords
    tsvdRatios = tsvd.explained_variance_ratio_

    total_variance = 0.0
    n_components = 0
    goal_var = 0.95


### MAX Features to components ###
#max:  2000  n:  1399
#max:  3000  n:  1891
#max:  4000  n:  2304
#max:  5000  n:  2598
#max:  6000  n:  2786
#max:  7000  n:  2807
#max:  8000  n:  2807
#max:  9000  n:  2807
#max:  10000  n:  2807

    for ratio in tsvdRatios:
        total_variance += ratio
        n_components += 1
    
        if total_variance >= goal_var:
            #print("Optimal Variance: ", total_variance, " with n components: ", n_components)
            break

    print("max: ", max, " n: ", n_components)

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
#CVLRModel = LogisticRegressionCV(n_jobs=-3, multi_class = 'multinomial', max_iter = 1000)
#CVLRModel.fit(bagOfWords, transLabels)
















