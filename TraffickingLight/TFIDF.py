from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_selection import chi2
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from time import time
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

import pandas as pd
import numpy as np
import os
import matplotlib.pylab as plt
from nltk.stem import WordNetLemmatizer
import re



path_to_csv = '/Users/Ju1y/Documents/Trafficking Light/Flagged/'
my_stop_words = text.ENGLISH_STOP_WORDS.union(['https'])
lemmatizer = WordNetLemmatizer()
dfT = pd.DataFrame(columns=['time stamp', 'user', 'full text', 'image url', 'Flag'])
dfF = pd.DataFrame(columns=['time stamp', 'user', 'full text', 'image url', 'Flag'])
dfA = pd.DataFrame(columns=['time stamp', 'user', 'full text', 'image url', 'Flag'])
#ngram_range=(lower bound number of words, upper bound number of words)
tfidf = TfidfVectorizer(ngram_range=(1,2), stop_words=my_stop_words, min_df=2, sublinear_tf=True)


#lemmatization method
def clean(tweet):

    cleaned_tweet = re.sub(r'([^\w\s]|\d)+', '', tweet.lower())
    return ' '.join([lemmatizer.lemmatize(i)
                for i in cleaned_tweet.split() if i not in my_stop_words])

#To form dataframe for matrix in preparation for scatter plot
def plotTFIDF(tfidf_dictionary):
    dfS = pd.DataFrame(columns=['word', 'IDF'])
    count = 0
    for key, value in tfidf_dictionary.items():
        dfS.loc[count] = [count + 1, value]
        count += 1

    # Scatterplot
    plt.scatter(x=dfS['word'], y=dfS['IDF'], c='DarkBlue')
    plt.xlabel('nth_feature', fontsize=16)
    plt.ylabel('IDF', fontsize=18)
    plt.title('IDF values of keywords')
    plt.savefig('IDF Graph.png')
    plt.show()


# This calculates the idf value for the terms in the posts and prints the highest ones
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

        dfT = dfT.append(csv_in[csv_in['Flag'] == 1]).astype('U13')
        dfF = dfF.append(csv_in[csv_in['Flag'] == 0]).astype('U13')
        dfA = dfA.append(csv_in)


print('Number of True sets: ', dfT['full text'].count())
print('Number of False sets: ', dfF['full text'].count())



#Unprocessed truth set fed into TFIDF
print(dfT['full text'])
response_true = tfidf.fit_transform(dfT['full text'])
feature_names = tfidf.get_feature_names()



#lemmatization going on here

def lemmatize(df):
    for row, index in zip(df['full text'], range(len(df['full text']))):
        row = clean(row)
        # cleans out the first 'b', any phrases that starts with https, and rt's
        row = re.sub(r'|(\b(https)\w*\s)|(\brt\s)', '', row)
        df['full text'].iloc[index] = row

lemmatize(dfT)

print('Cleaned Up')
print(dfT['full text'])

#lemmatized true tfidf set
response_true_lemmatized = tfidf.fit_transform(dfT['full text'])
feature_names = tfidf.get_feature_names()
tfidf_array = response_true_lemmatized.toarray()
print(tfidf_array)
print(feature_names)
print('count of feature names ', len(feature_names))
topTerms(tfidf)

#Analysis using Chi2

lemmatize(dfA)


response_all_lemmatized = tfidf.fit_transform(dfA['full text'])
tf_array = response_all_lemmatized.toarray()
top_n_terms = 60
features_chi2 = chi2(tf_array, dfA['Flag'] == 1)
indices = np.argsort(features_chi2[0])
feature_names = np.array(tfidf.get_feature_names())[indices]


unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-top_n_terms:])))
print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-top_n_terms:])))

feature_names = feature_names[::-1]

print(feature_names[:top_n_terms])


#-------------------------Machine Learing Model Development------------------------#
#create a svm classifier


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


np.set_printoptions(precision=2)

models = [
    RandomForestClassifier(n_estimators= 100, max_depth= 3),
    LinearSVC(penalty='l1', dual=False, tol=1e-3),
    MultinomialNB(alpha=.01),
    LogisticRegression(random_state=0, solver='lbfgs', multi_class='auto'),
    BernoulliNB(alpha=.01)
]


X = tfidf.transform(dfA['full text'])
y = np.asarray(dfA['Flag'], dtype="|S6")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25)

#oversampling
ros = RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(X_train, y_train)

print("n samples and n features: ", X_test.shape)

print('_' * 80)
print("Training: ")


#options: Multinomial NB, Linear SVC, SVC with different kernels, Bernoulli NB, Random Forest, Logistic Regression

df_unmarked = pd.DataFrame(columns= ['time stamp', 'user', 'full text', 'image url'])
unmarked_path = '/Users/Ju1y/Documents/Trafficking Light/S12/Training Data 19'
csv_in = pd.read_csv(unmarked_path+'.csv')

df_unmarked = df_unmarked.append(csv_in).astype('U13')


#cv_df = pd.DataFrame(columns=['model_name', 'time', 'accuracy'])
for model in models:
    model_name = model.__class__.__name__
    t0 = time()

    print('-----------', model_name, '----------------')
    clf = model

    clf.fit(X, y)
    train_time = time() - t0

    pred = clf.predict(tfidf.transform(df_unmarked['full text'])).astype('U13')
    df_unmarked.loc[:, model_name] = pred

    #pred = clf.predict(X_test)
    #test_time = time() - t0

csv_in = df_unmarked
csv_in.to_csv(unmarked_path+' Flagged.csv', index=False)

#Generates scores for confusion matrix
    #score = metrics.accuracy_score(y_test, pred)

    #cv_df.loc[len(cv_df)] =[model_name, test_time, score]

    #print("confusion matrix:")

    #print(metrics.confusion_matrix(y_test, pred))


    # Plot non-normalized confusion matrix
    #plot_confusion_matrix(y_test, pred, classes=['Non Trafficking', 'Trafficking'], title=('Confusion matrix, without normalization for '+ model_name))

    # Plot normalized confusion matrix
    #plot_confusion_matrix(y_test, pred, classes=['Non Trafficking', 'Trafficking'], normalize=True, title=('Normalized confusion matrix for '+ model_name))

    #plt.show()




#print(cv_df)






