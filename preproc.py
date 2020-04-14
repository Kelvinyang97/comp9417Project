import numpy as np
import pandas as pd
from collections import Counter
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



class preproc:
    
    def __init__(self, filename = '', tfidf = False, minmax = False, min_range = 1, 
                 max_range = 1, max_df = 1.0, min_df = 1, max_features = None):
        
        #Will transform features using the bag of words representation
        #Takes input:
        #string filename = location of data stored as csv
        #string process = either BAG_OF_WORDS for features or LABELLING for labels
        #boolean tfidf = True if term frequency inverse document frequency is to be used
        #   otherwise simple word count is used
        #boolean minmax = True if min-max normalisation is to be used. Can only be used
        #   if tfidf is False
        #Also takes CountVectoriser inputs if required:
        #string 'english' or list stop_words = list of stop words to be filtered 
        #   out of the bag. 'english' list has known issues
        #integer min_range = min number of n-grams to consider
        #integer max_range = max number of n-grams to consider
        #float in [0, 1.0] min_df = ignore features that have a frequency higher
        #   than this value
        #float in [0, 1.0] max_df = ignore features that have a frequency lower
        #   than this value
        #integer max_features = returns the highest n

        #Cleaning the reading process (only needs to happen once)
        
        featuresUnprocessed, labelsUnprocessed = self.read_data(filename)
        
        self.features = self.wordbag(featuresUnprocessed, tfidf = tfidf, minmax = minmax, min_range = min_range, max_range = max_range, max_df = max_df, min_df = min_df, max_features = max_features)
                                     
        self.labels = self.labelling(labelsUnprocessed)

    
    
    
    def read_data(self, filename):
        #Takes a csv file as input and outputs dataframes of features and labels
    
        #Read in data
        df = pd.read_csv(filename)
        temp_features_df = np.array(df.drop(['article_number','topic'], axis = 1))
        labels_df = df.drop(['article_number', 'article_words'], axis = 1).to_numpy()
    
        #Reformat features to be used in CountVectorizer
        features_df = [0] * len(temp_features_df)
        for i in range(len(temp_features_df)):
            features_df[i] = temp_features_df[i][0]
    
        return features_df, labels_df



    def wordbag(self, features, tfidf = False, minmax = False, min_range = 1,
                max_range = 1, max_df = 1.0, min_df = 1, max_features = None):
        #Takes a csv file as input along with CountVectoriser inputs and returns:
        #1. a bag of words as a scipy sparse matrix and
        #2. a list of all the words in the corpus
        
        #Transform corpus into a bag of words
        if tfidf:
            count = TfidfVectorizer(ngram_range = (min_range, max_range), max_df = max_df, 
                                    min_df = min_df, max_features = max_features)
        else:
            count = CountVectorizer(ngram_range = (min_range, max_range), max_df = max_df, 
                                    min_df = min_df, max_features = max_features)
                                    
        bag_of_words = count.fit_transform(features)
        
        #Scale bag of words using min max normalisation
        if not tfidf and minmax:
            scaled = preprocessing.MinMaxScaler()
            bag_of_words = scaled.fit_transform(bag_of_words.toarray())
            
        #Get a list of feature names
        feature_names = count.get_feature_names()
        
        return bag_of_words, feature_names
    
    
    
    def labelling(self, labels):
        #Takes a csv file as input and returns numpy arrays of:
        #1. transformed labels from strings to integers and 
        #2. unique list of topics in alphabetical order
        
        #Initialise label encoder
        le = preprocessing.LabelEncoder()
        
        #Fit and transform labels from strings to integers
        transformed_labels =le.fit_transform(labels.ravel())
        
        #Create a unique list of labels
        no_topics = len(Counter(transformed_labels).keys())
        list_of_labels = le.inverse_transform(range(no_topics))
        
        return transformed_labels, list_of_labels




