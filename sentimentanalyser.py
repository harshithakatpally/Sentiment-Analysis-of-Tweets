import os
import re
import time
import pickle
import itertools
import numpy as np
import matplotlib.pyplot as plot
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.metrics.classification import confusion_matrix
from textblob import TextBlob


#Count Total no.of tweets in each polarity category
def get_count(tweets_labelled):
    pos = neg = neut = 0
    for tweet in tweets_labelled:
        if tweet['sentiment'] == 'positive':
            pos += 1
        elif tweet['sentiment'] == 'negative':
            neg += 1
        elif tweet['sentiment'] == 'neutral':
            neut += 1
    print "\nPolarity of Tweets\n" + "Positive : " + str(pos),"\nNegative : " \
    + str(neg),"\nNeutral : " + str(neut) , "\n"


#Clean tweets using Regular Expression
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | \
    (\w +:\ / \ / \S +)", " ", tweet).split())


#Sense the Polarity
def get_sentiment(tweet):
    analysis = TextBlob(clean_tweet(tweet))
    if analysis.sentiment.polarity > 0:
        return 'positive'
    elif analysis.sentiment.polarity == 0:
        return 'neutral'
    else:
        return 'negative'


#Split data for crossvalidation with validation size = 0.20 and seed = 7
def cross_validate_data(tweets,tweet_labels):
    validation_size = 0.70
    seed = 7
    tweet_train, tweet_test, label_train, label_test = cross_validation.\
    train_test_split(tweets, tweet_labels, test_size = validation_size, random_state = seed)
    return tweet_train, tweet_test, label_train, label_test


#Plot Confusion Matrix for the classifier
def plot_confusion_matrix(cm, classes,title='Confusion Matrix',cmap=plot.cm.gist_ncar):
    plot.imshow(cm, interpolation='nearest', cmap=cmap)
    plot.title(title)
    plot.colorbar()
    tick_marks = np.arange(len(classes))
    plot.xticks(tick_marks, classes)
    plot.yticks(tick_marks, classes)
    plot.tight_layout()
    plot.ylabel('True label')
    plot.xlabel('Predicted label')

    #Label values on plotted gragh
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plot.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] < thresh else "black")


#Train data on classifier and plot confusion matrix
def show_report(train_vectors, label_train, test_vectors, label_test):
    classifier_rbf = svm.SVC(kernel='linear')
    print("Results for SVC(kernel=linear)")
    #Calculate Time taken for Training and Predicting Respectively
    t0 = time.time()
    classifier_rbf.fit(train_vectors, label_train)
    t1 = time.time()
    prediction_rbf = classifier_rbf.predict(test_vectors)
    t2 = time.time()
    time_rbf_train = t1 - t0
    time_rbf_predict = t2 - t1
    #Show Classification Report
    print("Training time: %fs; Prediction time: %fs" % (time_rbf_train, time_rbf_predict))
    print(classification_report(label_test, prediction_rbf))
    #Compute confusion matrix
    cnf_matrix = confusion_matrix(label_test, prediction_rbf)
    np.set_printoptions(precision = 2)
    print "Confusion Matrix\n",cnf_matrix
    #Plot non-normalized confusion matrix
    plot.figure()
    plot_confusion_matrix(cnf_matrix, classes = ['negative','neutral','positive'],
                          title = 'Confusion matrix')


#Vectorize the tweets to be trained on a classifier
def test_classifier(tweets,tweet_train, tweet_test, label_train, label_test):
    #Vectorize of cross validated data
    vectorizer = TfidfVectorizer(min_df=5,
                                 max_df=0.8,
                                 sublinear_tf=True,
                                 use_idf=True)
    train_vectors = vectorizer.fit_transform(tweet_train)
    test_vectors = vectorizer.transform(tweet_test)
    show_report(train_vectors, label_train, test_vectors, label_test)
    plot.show()


#Main method where tweets of a movie are retrieved for sentiment analysis and trained on a classifier
def main():
    tweets_labelled = []
    tweet_labels = []
    movie_index=1
    try:
        cwd = os.path.join(os.getcwd(), 'tweets')
        movies = os.listdir(cwd)
        print "List of Movie Datasets Available:"
        for movie_name in movies:
            print str(movie_index) + "." + movie_name
            movie_index += 1
        movie_name = input("Choose the movie:\t")
        movie_name = movies[movie_name - 1]
        print "Selected Dataset: " + movie_name
        cwd = os.path.join(cwd, movie_name)
        with open(cwd, 'r') as f:
            tweets = f.readlines()
            tweets = tweets[:len(tweets):2]
            for tweet in tweets:
                label = get_sentiment(tweet)
                tweets_labelled.append({'text':tweet,'sentiment':label})
                tweet_labels.append(label)
            get_count(tweets_labelled)
            tweet_train, tweet_test, label_train, label_test = \
            cross_validate_data(tweets,tweet_labels)
            test_classifier(tweets,tweet_train, tweet_test, label_train, label_test)
    except BaseException  as e:
        print e

if __name__ == "__main__":
    main()
