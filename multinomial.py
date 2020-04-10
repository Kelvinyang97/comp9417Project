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
from sklearn import svm


#Read in data
df = pd.read_csv('training.csv')
temp_features_df = np.array(df.drop(['article_number','topic'], axis = 1))
labels_df = df.drop(['article_number', 'article_words'], axis = 1).to_numpy()


#Reformat features to be used in CountVectorizer
features_df = [0] * len(temp_features_df)
for i in range(len(temp_features_df)):
    features_df[i] = temp_features_df[i][0]


#Transform corpus into a bag of words
count = CountVectorizer()
bag_of_words = count.fit_transform(features_df)
array_bag_of_words = bag_of_words.toarray()
feature_names = count.get_feature_names()


#Transform labels
le = preprocessing.LabelEncoder()
le.fit(labels_df)
transformed_labels = le.transform(labels_df.ravel())
no_topics = len(Counter(transformed_labels).keys())
labels = le.inverse_transform(range(no_topics))

def getAP(a,d,n): 
    res = []
    curr_term = a
    res.append(a)
    for i in range(1,n):
        curr_term =curr_term + d 
        res.append(curr_term)
    return res

x_train = bag_of_words
y_train = transformed_labels
# from sklearn.model_selection import RandomizedSearchCV

###tuning parameter alpha
###random search to find a rough good region to explore for alpha
# mnbc = MultinomialNB()
# alpha_range = getAP(0.2, 0.2, 200)
# random_grid = {'alpha': alpha_range}
# random_search = RandomizedSearchCV(estimator=mnbc,
#                                    param_distributions=random_grid,
#                                    n_iter=50,
#                                    scoring='accuracy',
#                                    cv=3)
# random_search.fit(x_train, y_train)
# print(random_search.best_params_)

from sklearn.model_selection import GridSearchCV
###grid search to find the best region for alpha
mnbc = MultinomialNB()
alpha_range = getAP(0.1, 0.01, 1000)
param_grid = {'alpha': alpha_range}
grid_search = GridSearchCV( estimator=mnbc,
                            param_grid=param_grid,
                            scoring='accuracy',
                            cv=3)
grid_search.fit(x_train, y_train)
print(grid_search.best_params_)
best_mnbc = grid_search.best_estimator_

scores = cross_val_score(best_mnbc, x_train, y_train, cv = 5)
print(scores)
print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores) * 100),'%')


#can refer to this link:
#https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/04.%20Model%20Training/09.%20MT%20-%20MultinomialNB.ipynb
