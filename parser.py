__version__ = '0.1.0'

import numpy as np
import csv


class Parse:

    TRAINCSV = "data/training.csv"
    TESTCSV = "data/test.csv"
    CSVKEYS = ['No', 'Words', 'Topic']

    ### FIRST STRUCTURE ###
    # Dictionary of each column
    # List of article numbers
    # List of dictionaries containing word counts
    # List of the identified/correct topics
    # Training = {No. = [], words = [dicts()], topic = []}
    
    ### SECOND STRUCTURE ###
    # List (row) of List (column)
    
    ### THIRD STRUCTURE ###
    # Dictionary of article No.
    # each entry is a List storing = [ Dict{ Words }, topic ]
    
    ### FOURTH STRUCTURE ###
    # Dictionary of article No.
    # each entry is a List storing = [ [words], topic ]
    
    def __init__(self, style=1):
        if style == 1:
            self.trainSet = self.style1(self.TRAINCSV)
            self.testSet = self.style1(self.TESTCSV)
        elif style == 2:
            self.trainSet = self.style2(self.TRAINCSV)
            self.testSet = self.style2(self.TESTCSV)
        elif style == 3:
            self.trainSet = self.style3(self.TRAINCSV)
            self.testSet = self.style3(self.TESTCSV)
        elif style == 4:
            self.trainSet = self.style4(self.TRAINCSV)
            self.testSet = self.style4(self.TESTCSV)
        else:
            print("ERROR: Please give a style number")
    
    
    ################
    ### METHODS ####
    ################
    
    
    ### GETTERS ###
    def getTrain(self):
        return self.trainSet
        
    def getTest(self):
        return self.testSet
        
    
    
    # words: A list of words
    # Output: A dictionary, performing a frequency count of each word
    def buildWordDict(self, words):
        freqCount = dict()
        for entry in words:
            if entry in freqCount:
                freqCount[entry] += 1
            else:
                freqCount[entry] = 1
        return freqCount
    
    
    #### STRUCTURAL PARSERS BY INIT DEFINITION 1 -> 4 ###
    
    #Dict of columns, Words as Dict
    def style1(self, fileName):
        with open(fileName, newline='') as f:
            next(f) #skip column names
            preformat = csv.DictReader(f, self.CSVKEYS)
            
            #Build Column Dictionary => Will always be first time
            Set = {'No':[], 'Words':[], 'Topic':[]}
            for row in preformat:
                words = row['Words'].split(',')
                wordDict = self.buildWordDict(words)
                
                Set['No'].append(int(row['No']))
                Set['Words'].append(wordDict)
                Set['Topic'].append(row['Topic'])
            
            return Set
           
    #List of Lists
    def style2(self, fileName):
        with open(fileName, newline='') as f:
            next(f)
            preformat = csv.DictReader(f, self.CSVKEYS)
            
            #create column lists
            No = []
            Words = []
            Topic = []
            
            #append to columns by row
            for row in preformat:
                No.append(int(row['No']))
                Words.append(row['Words'].split(','))
                Topic.append(row['Topic'])
            
            #Push into a List of Lists
            #Access elems by selecting column then row, eg: arr2D[1][500], fifth hundred entry, containing list of words
            return [No, Words, Topic]
            
            
    #Article No Dict (always from zero), Words as Dict
    def style3(self, fileName):
        with open(fileName, newline='') as f:
            next(f)
            preformat = csv.DictReader(f, self.CSVKEYS)
            
            #Create a dictionary to store entries by row
            Set = {}
            init = 0
            for row in preformat:
                if init < 1:
                    init += 1
                    offset = int(row['No'])
                ArticleNo = int(row['No']) - offset
                if ArticleNo in Set:
                    continue
                else:
                    words = row['Words'].split(',')
                    wordDict = self.buildWordDict(words)
                    Set[ArticleNo] = [wordDict, row['Topic']]
                    
            return Set
            
    #Article No Dict (always from zero), Words as List
    def style4(self, fileName):
        with open(fileName, newline='') as f:
            next(f)
            preformat = csv.DictReader(f, self.CSVKEYS)
        
            #Create a dictionary to store entries by row
            Set = {}
            init = 0
            for row in preformat:
                if init < 1:
                    init += 1
                    offset = int(row['No'])
                ArticleNo = int(row['No']) - offset
                if ArticleNo in Set:
                    continue
                else:
                    words = row['Words'].split(',')
                    Set[ArticleNo] = [words, row['Topic']]
                
            return Set


##############################
### END STRUCTURAL PARSERS ###
##############################


