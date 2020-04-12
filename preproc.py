import numpy as np
import pandas as pd
from collections import Counter
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



class preproc:
    
    def __init__(self, filename = '', process = '', tfidf = False, stop_words = None, 
                 min_range = 1, max_range = 1, max_df = 1, min_df = 1, max_features = None):
        
        #Will transform features using the bag of words representation
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
        if process == 'BAG_OF_WORDS':
            print('Bag of Words process requested')
            self.features = self.wordbag(filename, tfidf = tfidf, stop_words = stop_words, 
                                         min_range = min_range, max_range = max_range, 
                                         max_df = max_df, min_df = min_df, max_features = max_features)
        
        #Will transform features using min-max normalisation
        elif process == 'MIN_MAX':
            print('Min-max normalisation requested')
            self.features = self.minmax(filename)
        
        #Will relabel string categories into integers
        elif process == 'LABELLING':
            print('Re-labelling categories requested')
            self.labels = self.labelling(filename)
        
        #Error catch for unspecified process
        else:
            print('==========================')
            print('Invalid process requested!')
            print('==========================')
    
    
    
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


    def wordbag(self, filename, tfidf = False, stop_words = None, min_range = 1, 
                max_range = 1, max_df = 1, min_df = 1, max_features = None):
        #Takes a csv file as input along with CountVectoriser inputs and returns:
        #1. a bag of words as a scipy sparse matrix and
        #2. a list of all the words in the corpus
        
        #Read data in
        features, labels = self.read_data(filename)
        
        #Transform corpus into a bag of words
        if tfidf:
            count = TfidfVectorizer(stop_words = stop_words, 
                                    ngram_range = (min_range, max_range), max_df = max_df, 
                                    min_df = min_df, max_features = max_features)
        else:
            count = CountVectorizer(stop_words = stop_words, 
                                    ngram_range = (min_range, max_range), max_df = max_df, 
                                    min_df = min_df, max_features = max_features)
        bag_of_words = count.fit_transform(features)

        #Get a list of feature names
        feature_names = count.get_feature_names()
        
        return bag_of_words, feature_names
    
    
    def minmax(self, filename):
        #Takes a csv file as input and returns;
        #1. a numpy array of min-max scaled features
        
        #Create a bag of words
        bag_of_words, feature_names = self.wordbag(filename)
        
        #Initialise min-max scaler
        scaled = preprocessing.MinMaxScaler()
        
        #Scale bag of words using min max normalisation
        minmax_bag_of_words = scaled.fit_transform(bag_of_words.toarray())
        
        return minmax_bag_of_words
    
    def labelling(self, filename):
        #Takes a csv file as input and returns numpy arrays of:
        #1. transformed labels from strings to integers and 
        #2. unique list of topics in alphabetical order
        
        #Read data in
        features, labels = self.read_data(filename)
        
        #Initialise label encoder
        le = preprocessing.LabelEncoder()
        
        #Fit and transform labels from strings to integers
        transformed_labels =le.fit_transform(labels.ravel())
        
        #Create a unique list of labels
        no_topics = len(Counter(transformed_labels).keys())
        list_of_labels = le.inverse_transform(range(no_topics))
        
        return transformed_labels, list_of_labels



TRAINCSV = 'training.csv'
TESTCSV = 'test.csv'
    
p = preproc(TRAINCSV, 'BAG_OF_WORDS', min_range = 2, max_range= 2)
print(p.features[0])
print(p.features[1])
