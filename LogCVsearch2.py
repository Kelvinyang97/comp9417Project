import matplotlib.pyplot as plt
import numpy as np
import output
import joblib
from preproc import preproc as Parse

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.decomposition import TruncatedSVD


#Get Access to data
DATA = Parse('data/', tfidf=True, min_df = 10, max_features = 7000)
#Assign features
bagOfWords = DATA.features[0]
testBag = DATA.features[1]
bagNames = DATA.features[2]

#Scale
transformer = TruncatedSVD(n_components = 2807)
transformer.fit(bagOfWords)
bagOfWords = transformer.transform(bagOfWords)
testBag = transformer.transform(testBag)

#Assign Labels
transLabels = DATA.labels[0]
testLabels = DATA.labels[1]
listLabels = DATA.labels[2]


### SET UP GRID SEARCH ###
#estimator
#LR = LogisticRegression()

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
# refit = T
# intercept_scaling = 1.0
# multi_class = 'auto' --> Ensure multiple classes with multinomial
# random_state = None (good for ensure consistent build)
# l1_ratios = None
Cs = np.logspace(-20, 20, 100)

try:
    with open("sciModels/CV2.model", 'rb') as f:
        CV2 = joblib.load(f)
        
        #### FIRST MODEL ####
        print("1:LBFGS")
        #print("C_: ")
        #print(CV2.C_)
        #print("Scores_")
        #print(CV2.scores_)
        print("Train: ", CV2.score(bagOfWords, transLabels))
        print("Test: ", CV2.score(testBag, testLabels))
        
        actual = testLabels
        pred = CV2.predict(testBag)
        pred_probs = CV2.predict_proba(testBag)

        output.output_results(actual, pred, pred_probs, listLabels)
        
except EnvironmentError:
    print("Made it here!")
    CV2 = LogisticRegressionCV(n_jobs=-1, Cs=Cs, solver='lbfgs', multi_class = 'multinomial', max_iter = 5000)
    CV2.fit(bagOfWords, transLabels)
    
    #### SECOND MODEL ####
    print("2:LBFGS")
    #print("Cs_: ")
    #print(CV2.Cs_)
    #print("Scores_")
    #print(CV2.scores_)
    #print("classes_")
    #print(CV2.classes_)
    print("Train: ", CV2.score(bagOfWords, transLabels))
    print("Test: ", CV2.score(testBag, testLabels))
    
    actual = testLabels
    pred = CV2.predict(testBag)
    pred_probs = CV2.predict_proba(testBag)

    output.output_results(actual, pred, pred_probs, listLabels)
    
    with open("sciModels/CV2.model", 'wb') as f:
        joblib.dump(CV2, f)
        print("Wrote file to sciModels/CV2.model")












