from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import text
from sklearn import svm
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
import nltk
from nltk.stem import WordNetLemmatizer
import re



path_to_csv = '/Users/Ju1y/Documents/Trafficking Light/Flagged/'
my_stop_words = text.ENGLISH_STOP_WORDS.union(['https'])
lemmatizer = WordNetLemmatizer()
dfT = pd.DataFrame(columns=['time stamp', 'user', 'full text', 'image url', 'Flag'])
dfF = pd.DataFrame(columns=['time stamp', 'user', 'full text', 'image url', 'Flag'])
#ngram_range=(lower bound number of words, upper bound number of words)
tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words=my_stop_words, norm='l2', min_df=2)


#lemmatization method
def clean(tweet):

    cleaned_tweet = re.sub(r'([^\w\s]|\d)+', '', tweet.lower())
    return ' '.join([lemmatizer.lemmatize(i, 'v')
                for i in cleaned_tweet.split() if i not in my_stop_words])

#To form dataframe for matrix in preparation for scatter plot
def plotTFIDF(tfidf_dictionary):
    dfS = pd.DataFrame(columns=['word', 'TFIDF'])
    count = 0
    for key, value in tfidf_dictionary.items():
        dfS.loc[count] = [count + 1, value]
        count += 1

    # Scatterplot
    plt.scatter(x=dfS['word'], y=dfS['TFIDF'], c='DarkBlue')
    plt.xlabel('words', fontsize=16)
    plt.ylabel('TFIDF', fontsize=18)
    plt.title('TFIDF values of keywords')
    plt.savefig('TFIDF Graph.png')
    plt.show()


def topTerms(vectorizer):
    indices = np.argsort(vectorizer.idf_)[::-1]
    top_n = 40
    top_features = dict(zip([feature_names[i] for i in indices[:]], [vectorizer.idf_[i] for i in indices[:]]))
    print(top_features)

    plotTFIDF(top_features);

#imports csv and separates datasets into True and False sets
for pos_csv in os.listdir(path_to_csv):
    if pos_csv.endswith('.csv'):
        csv_in = pd.read_csv(path_to_csv + pos_csv)

        dfT = dfT.append(csv_in[csv_in['Flag'] == 'Y'])
        dfF = dfF.append(csv_in[csv_in['Flag'] == 'N'])

print('Number of True sets: ', dfT['full text'].count())
print('Number of False sets: ', dfF['full text'].count())

#Unprocessed truth set fed into TFIDF
print(dfT['full text'])
response = tfidf.fit_transform(dfT['full text'])
feature_names = tfidf.get_feature_names()
topTerms(tfidf)


#lemmatization going on here

for row, index in zip(dfT['full text'], range(len(dfT['full text']))):
    row = clean(row)
    #cleans out the first 'b', any phrases that starts with https, and rt's
    row = re.sub(r'(^b)|(\b(https)\w*\s)|(\brt\s)', '', row)
    dfT['full text'].iloc[index] = row

print('Cleaned Up')
print(dfT['full text'])

#lemmatized tfidf set
response_lemmatized = tfidf.fit_transform(dfT['full text'])
feature_names = tfidf.get_feature_names()
tfidf_array = response_lemmatized.toarray()
print(tfidf_array)
print(feature_names)
print('count of feature names ', len(feature_names))
topTerms(tfidf)

# This calculates the idf value for the terms in the posts and prints the highest ones


#-------------------------Machine Learing Model Development------------------------#
#create a svm classifier
clf = svm.SVC(kernel= 'linear')









