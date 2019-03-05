from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.feature_extraction import text
import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt



path_to_csv = '/Users/Ju1y/Documents/Trafficking Light/Flagged/'
my_stop_words = text.ENGLISH_STOP_WORDS.union(['https'])

dfT = pd.DataFrame(columns=['time stamp', 'user', 'full text', 'image url', 'Flag'])
dfF = pd.DataFrame(columns=['time stamp', 'user', 'full text', 'image url', 'Flag'])

for pos_csv in os.listdir(path_to_csv):
    if pos_csv.endswith('.csv'):
        csv_in = pd.read_csv(path_to_csv + pos_csv)

        dfT = dfT.append(csv_in[csv_in['Flag'] == 'Y'])
        dfF = dfF.append(csv_in[csv_in['Flag'] == 'N'])


#ngram_range=(lower bound number of words, upper bound number of words)

print(dfT['full text'].count())
print(dfF['full text'].count())
print(dfT[dfT['full text'].str.contains('gangbang')].count())
tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words=my_stop_words, norm='l2', min_df=2)
response = tfidf.fit_transform(dfT['full text'])

feature_names = tfidf.get_feature_names()
tfidf_array = response.toarray()
print('count of feature names ', len(feature_names))
print(dfT['full text'])
print(tfidf_array)
print("2")

# This calculates the idf value for the terms in the posts and prints the highest ones
indices = np.argsort(tfidf.idf_)[::-1]

top_n = 40
top_features = dict(zip([feature_names[i] for i in indices[:]],[tfidf.idf_[i] for i in indices[:]]))
print (top_features)

#To form dataframe for matrix in preparation for scatter plot
dfS = pd.DataFrame(columns=['word', 'TFIDF'])

count = 0

for key,value in top_features.items():
    dfS.loc[count] = [count+1,value]
    count += 1

print(dfS)

#Scatterplot
plt.scatter(x=dfS['word'], y=dfS['TFIDF'], c='DarkBlue')
plt.xlabel('words', fontsize=16)
plt.ylabel('TFIDF', fontsize=18)
plt.title('TFIDF values of keywords')
plt.savefig('TFIDF Graph.png')
plt.show()




#


"""doc = 0
for col in response.nonzero()[1]:
    print (feature_names[col], ' - ', response[doc, col])"""







