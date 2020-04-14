import numpy as np
import matplotlib.pyplot as plt
from preproc import preproc


def sum_stats(features, feature_names, labels, label_names, highest = 10, tfidf = False):
    #Takes features, feature names, labels and label names and prints
    #summary statistics to the terminal.
    
    #For data that has been normalised by term frequency inverse document frequency
    if tfidf:
        for i in range(len(label_names)):
            print('========================')
            print('The ', highest, ' words with the highest TF-IDF score across ', label_names[i], ' articles are:', sep='')
        
        
        
    #For data that hasn't been normalised by term frequency inverse document frequency
    else:
        #Group articles by topic
        array_features = features.toarray()
        no_words = array_features.shape[1]
        total = np.zeros((len(label_names), no_words))
        for i in range(len(labels)):
            total[labels[i]] = total[labels[i]] + array_features[i]
        
        
        #Print summary statistics for corpus...
        print('There are ', '{:,}'.format(int(np.sum(total))), ' words in the corpus.', sep='')
        print('There are ', '{:,}'.format(len(feature_names)), ' unique words in the corpus.', sep='')
        
        
        #...then foreach topic in corpus
        for i in range(len(label_names)):
            
            no_words = np.sum(total[i])
            unique_words = np.count_nonzero(total[i])
            freq_words = sorted(zip(total[i], feature_names), reverse = True)[:highest]
            
            #Summary statistics for each topic...
            print('========================')
            print('The total number of words in', label_names[i], 'articles is: ', '{:,}'.format(int(no_words)))
            print('The total number of unique words in', label_names[i], 'articles is: ', '{:,}'.format(int(unique_words)))
            print('The ', highest, ' most frequent words are:', sep='')
            
            
            #... and most frequent words
            for j in range(len(freq_words)):
                
                print(freq_words[j][1], ' ' * (20 - len((freq_words[j][1]))), 
                      '{:,}'.format(int(freq_words[j][0])), 
                      ' ' * (8 - len('{:,}'.format(int(freq_words[j][0])))), 
                      '(', '{:.2f}'.format(freq_words[j][0] / no_words * 100),'%)', sep='')
    
    return



def article_freq(labels, label_names, colours = ["#C00000", "#FF0000", "#FFC000", "#FFFF00", "#92D050", 
           "#00B050", "#00B0F0", "#0070C0", "#7030A0", "#000000", "#BFBFBF"]):
    #Takes labels and label names and outputs the distribution of articles
    #by topic in both print and graph form.
    
    
    article_count = []
    total = len(labels)
    print('========================')
    print('Article Distribution (%)')
    print('========================')
    
    
    #Count number of occurrences for each topic
    for i in range(len(label_names)):
    
        n = np.count_nonzero(labels == i)
        freq = n / total
        article_count.append(freq)
        print(label_names[i], ' ' * (40 - len(label_names[i])), '{:.2f}'.format(freq * 100), '%', sep = '')
    
    
    #Plot article distribution
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(label_names, article_count, color = colours)
    plt.xticks(rotation = 90)
    plt.show()
    
    return






TRAINCSV = 'training.csv'
TESTCSV = 'test.csv'
colours = ["#C00000", "#FF0000", "#FFC000", "#FFFF00", "#92D050", 
           "#00B050", "#00B0F0", "#0070C0", "#7030A0", "#000000", "#BFBFBF"]

#Preprocess features and labels 
f = preproc(TRAINCSV, 'BAG_OF_WORDS', tfidf = True)
l = preproc(TRAINCSV, 'LABELLING')

#Store output into variables
train_features = f.features[0]
feature_names = f.features[1]
train_labels = l.labels[0]
list_of_labels = l.labels[1]

print(train_features)

sum_stats(train_features, feature_names, train_labels, list_of_labels, tfidf = True)
#article_freq(train_labels, list_of_labels)