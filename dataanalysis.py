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
df = pd.read_csv('data/training.csv')
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


#Explanatory Analysis
#Group articles by topics
no_words = array_bag_of_words.shape[1]
total = np.zeros((no_topics, no_words))
for i in range(len(transformed_labels)):
    total[transformed_labels[i]] = total[transformed_labels[i]] + array_bag_of_words[i]


#Print summary statistics
print('There are ', '{:,}'.format(int(np.sum(total))), ' words in the corpus.', sep='')
print('There are ', '{:,}'.format(len(feature_names)), ' unique words in the corpus.', sep='')
highest = 10
for i in range(no_topics):
    no_words = np.sum(total[i])
    unique_words = np.count_nonzero(total[i])
    print('========================')
    print('The total number of words in', labels[i], 'articles is: ', '{:,}'.format(int(no_words)))
    print('The total number of unique words in', labels[i], 'articles is: ', '{:,}'.format(int(unique_words)))
    freq_words = sorted(zip(total[i], feature_names), reverse = True)[:highest]
    print('The ', highest, ' most frequent words are:', sep='')
    for j in range(len(freq_words)):
        print(freq_words[j][1], ' ' * (20 - len((freq_words[j][1]))), 
              '{:,}'.format(int(freq_words[j][0])), 
              ' ' * (8 - len('{:,}'.format(int(freq_words[j][0])))), 
              '(', '{:.2f}'.format(freq_words[j][0] / no_words * 100),'%)', sep='')


#Keep irrelevant articles
print('========================')
print('    ALL ARTICLES  (%)   ')
print('========================')
no_articles = df['topic'].value_counts(normalize = True).sort_index(ascending = True) * 100
print(no_articles)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
types = no_articles.index
total = no_articles
ax.bar(types, total)
plt.xticks(rotation = 90)
plt.show()


#Remove irrelevant articles
print('========================')
print(' RELEVANT ARTICLES   (%)')
print('========================')
df_rel = df[df.topic != 'IRRELEVANT']
no_articles_rel = df_rel['topic'].value_counts(normalize = True).sort_index(ascending = True) * 100
print(no_articles_rel)
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
types = no_articles_rel.index
total = no_articles_rel
ax.bar(types, total)
plt.xticks(rotation = 90)
plt.show()



#ML Methods
#Logistic Regression
x_train = bag_of_words
y_train = transformed_labels
scores = cross_val_score(LogisticRegression(max_iter=50), x_train, y_train, cv = 5)
print(scores)
print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores) * 100),'%')

mnbc = MultinomialNB()
scores = cross_val_score(mnbc, x_train, y_train, cv = 5)
print(scores)
print("Mean cross-validation accuracy: {:.2f}".format(np.mean(scores) * 100),'%')


#can refer to this link:
#https://github.com/miguelfzafra/Latest-News-Classifier/blob/master/0.%20Latest%20News%20Classifier/04.%20Model%20Training/09.%20MT%20-%20MultinomialNB.ipynb
