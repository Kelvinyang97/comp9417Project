from math import floor, ceil
from preproc import preproc
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression


def output_results(actual, predicted, probabilities, label_names, threshold = 0.5, extra_output = True):
    #Takes as inputs a numpy array of actual fitted labels, predited labels, 
    #class probabilities, label names plus a float for prediction threshold,
    #and a boolean to print out extra output if wanted. The output is printed
    #to the terminal in the specified table format.
    
    
    #Initialise variables
    precision = []
    recall = []
    f1 = []
    max_probs = []
    top_10_articles = [[] for i in range(len(label_names))]
    
    
    #Create confusion matrix
    matrix = confusion_matrix(actual, predicted)
    
    
    #Calculate precision, recall and f1 scores
    for i in range(len(label_names)):
        tp = matrix[i][i]
        tpfp = sum(matrix[i])
        tpfn = [sum(row[i] for row in matrix)][0]
        precision.append(tp / tpfp)
        recall.append(tp / tpfn)
        f1.append((2 * tp / tpfp * tp / tpfn) / (tp / tpfp + tp / tpfn))


    #Determine recommended articles
    #Calculate most likely prediction
    for i in range(len(probabilities)):
        
        max_value = max(probabilities[i])
        pred_class = probabilities[i].argmax()
        max_probs.append([max_value, pred_class, i + 1])
    
    #Sort based on the highest to lowest values
    max_probs = sorted(max_probs, key = lambda x: x[0])[::-1]

    #Get the 10 most relevant articles per topic
    for i in range(len(max_probs)):
        prob = max_probs[i][0]
        topic = max_probs[i][1]
        if len(top_10_articles[topic]) < 10 and prob >= threshold:
            top_10_articles[topic].append(max_probs[i])
    print(top_10_articles)
    
    #Print results
    #Print headers
    print('=' * 115)
    print('||  TOPIC NAME', ' ' * (40 - len('TOPIC NAME')), '|| SUGGESTED ARTICLES ||  PRECISION  ||    RECALL   ||      F1     ||')
    print('=' * 115)
    
    #Loop through all topics
    for i in range(len(label_names)):
        
        #Disregard irrelevant articles
        if label_names[i] != 'IRRELEVANT':
            
            #Create formated variables to aid in printing
            format_precision = '{:.2f}'.format(precision[i] * 100) + '%'
            format_recall = '{:.2f}'.format(recall[i] * 100) + '%'
            format_f1 = '{:.2f}'.format(f1[i] * 100) + '%'
            format_article = '{:,}'.format(top_10_articles[i][0][2])
            len_fp = len(format_precision)
            len_fr = len(format_recall)
            len_ff = len(format_f1)
            len_fa = len(format_article)
            pre_fp = ceil((13 - len_fp) / 2)
            post_fp = floor((13 - len_fp) / 2)
            pre_fr = ceil((13 - len_fr) / 2)
            post_fr = floor((13 - len_fr) / 2)
            pre_ff = ceil((13 - len_ff) / 2)
            post_ff = floor((13 - len_ff) / 2)
            pre_fa = ceil((20 - len_fa) / 2)
            post_fa = floor((20 - len_fa) / 2)
            
            #Print first line of output
            print('||  ', label_names[i], ' ' * (42 - len(label_names[i])), 
                  '||', ' ' * pre_fa, format_article, ' ' * post_fa,
                  '||', ' ' * pre_fp, format_precision, ' ' * post_fp,
                  '||', ' ' * pre_fr, format_recall, ' ' * post_fr,
                  '||', ' ' * pre_ff, format_f1, ' ' * post_ff, '||', sep = '')
            
            #Print remaining suggested articles
            if len(top_10_articles[i]) > 1:
                for j in range(1, len(top_10_articles[i])):
                    format_article = '{:,}'.format(top_10_articles[i][j][2])
                    len_fa = len(format_article)
                    pre_fa = ceil((20 - len_fa) / 2)
                    post_fa = floor((20 - len_fa) / 2)
                    print('||  ', ' ' * 42, 
                          '||', ' ' * pre_fa, format_article, ' ' * post_fa,
                          '||', ' ' * 13, '||', ' ' * 13, 
                          '||', ' ' * 13, '||', sep = '')
            print('=' * 115)
    
    
    #Print incorrectly suggested articles if user requires them
    if extra_output:
        print('\n')
        print('Incorrectly predicted suggestions:')
        for i in range(len(top_10_articles)):
            for j in range(len(top_10_articles[i])):
                suggested_article_pred_prob = '{:.2f}'.format(top_10_articles[i][j][0] * 100) + '%'
                pred_class = top_10_articles[i][j][1]
                article_number = top_10_articles[i][j][2]
                actual_class = actual[article_number - 1]
                actual_article_pred_prob = '{:.2f}'.format(probabilities[article_number - 1][actual_class] * 100) + '%'
                if pred_class != actual_class:
                    print('Article ', article_number, ' incorrectly suggested.', sep = '')
                    print('Predicted ', label_names[pred_class], ' with ', suggested_article_pred_prob, 
                          '. Actual class is ', label_names[actual_class], ' with probability ', actual_article_pred_prob, sep = '')
    
    return


TRAINCSV = 'training.csv'
TESTCSV = 'test.csv'

#Preprocess features and labels 
f = preproc(filepath = '', trainname = TRAINCSV, testname = TESTCSV, tfidf = True, max_range = 2, min_df = 10, max_features = 1000)
    
#Store output into variables
train_features = f.features[0]
feature_names = f.features[2]
train_labels = f.labels[0]
list_of_labels = f.labels[2]

#Run logistic regression
#Note this can be changed to any model
model1 = LogisticRegression(random_state=0, max_iter = 250).fit(train_features, train_labels)
pred = model1.predict(train_features)
pred_probs = model1.predict_proba(train_features)

#Output results
output_results(train_labels, pred, pred_probs, list_of_labels, threshold = 0.2)
