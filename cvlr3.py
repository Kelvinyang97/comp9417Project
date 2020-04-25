import matplotlib.pyplot as plt
import numpy as np
import output
import joblib
from preproc import preproc as Parse


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn import preprocessing
from sklearn.decomposition import TruncatedSVD

#Get Access to data
DATA = Parse('data/', tfidf=True, min_df = 10, max_features = 7000 )

#Assign features
bagOfWords = DATA.features[0]
testBag = DATA.features[1]
bagNames = DATA.features[2]

#AttemptScaling
#n_iter = 10
transformer = TruncatedSVD(n_components = 2807)
transformer.fit(bagOfWords)
bagOfWords = transformer.transform(bagOfWords)
testBag = transformer.transform(testBag)


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
try:
    with open("sciModels/CVLR3.model", 'rb') as f:
        
        CVLRModel = joblib.load(f)
        
        print( 'Score on training data: ', CVLRModel.score(bagOfWords, transLabels) )
        print( 'Score on test data: ', CVLRModel.score(testBag, testLabels) )


        actual = testLabels
        pred = CVLRModel.predict(testBag)
        pred_probs = CVLRModel.predict_proba(testBag)

        output.output_results(actual, pred, pred_probs, listLabels)
        
        
except EnvironmentError:

    CVLRModel = LogisticRegressionCV( n_jobs=-3, multi_class = 'multinomial', max_iter = 2000)
    CVLRModel.fit(bagOfWords, transLabels)
    
    
    print( 'Score on training data: ', CVLRModel.score(bagOfWords, transLabels) )
    print( 'Score on test data: ', CVLRModel.score(testBag, testLabels) )

    actual = testLabels
    pred = CVLRModel.predict(testBag)
    pred_probs = CVLRModel.predict_proba(testBag)

    output.output_results(actual, pred, pred_probs, listLabels)
    
    with open("sciModels/CVLR3.model", 'wb') as f:
        joblib.dump(CVLRModel, f)
        print("Wrote file to sciModels/CVLR3.model")

