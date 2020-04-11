# Build logistic regression model, using training set 


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
#from sklearn.feature_extraction.text import TfidVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from collections import Counter


# maybe abstract this later as it currently being repeated alot across files
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

x_train = bag_of_words
y_train = transformed_labels


# tune: regularization strength
parameters = {'C': [0.001, 0.01, 0.1, 1, 5, 10]} 
# test on grid search 
grid_search = GridSearchCV(LogisticRegression(), param_grid = parameters, scoring = 'accuracy', cv = 3)
estimator = grid_search.fit(x_train, y_train)
best_reg = estimator.best_params_['C']
best_regression = LogisticRegression(C = best_reg)
print(f"Best regularization strength, grid search: {best_reg}")
        
# get the cross val score after this tuning
score = cross_val_score(best_regression, x_train, y_train, cv = 5)
print("Mean cross-validation accuracy for bag of words: {:.2f}".format(np.mean(score) * 100),'%')
     
# test on randomised search
randomised_search = RandomizedSearchCV(LogisticRegression(), parameters, scoring = 'accuracy', cv = 3)
random_estimator = randomised_search.fit(x_train, y_train)
best_rand_reg =  random_estimator.best_params_['C']
print(f"Best regularization strength, randomized search: {best_rand_reg}") 

# tune  tf_idf score -> method followed from pg 337 of intro to ML
'''
def tuneTFIDF(self, x_train, y_train):
pipe = make_pipeline(TfidfVectorizer(min_df = 5), LogisticRegression())
grid = GridSearchCV(pipe) # don't know if i should also put C in 
grid.fit(x_train, y_train)
cv_score = grid.best_score_
print(f"Validation score after tf_idf : {cv_score}") 
'''       
# tune min_df-> minimum number of docs a word needs to appear 
# if this doesn't affect accuracy it can aid processesing speed
 

# tune cv size think this may have to be done iteratively ? 


