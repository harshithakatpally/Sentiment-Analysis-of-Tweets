import os
import re
import numpy as np
import matplotlib.pyplot as plot
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
    return pos,neg,neut


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


#Calculating the polarity count for all movies
def calculate_polarity_count_of_tweets():
    positive_count = []
    negative_count = []
    neutral_count = []
    tweets_labelled = []
    tweet_labels = []
    try:
        cwd = os.path.join(os.getcwd(), 'tweets')
        movies = os.listdir(cwd)
        for movie_name in movies:
            with open(os.path.join(cwd, movie_name), 'rb') as f:
                tweets = f.readlines()
                tweets = tweets[:len(tweets):2]
                for tweet in tweets:
                    tweet = tweet.decode(errors = 'replace')
                    label = get_sentiment(tweet)
                    tweets_labelled.append({'text': tweet, 'sentiment': label})
                    tweet_labels.append(label)
                pos, neg, neut = get_count(tweets_labelled)
                positive_count.append(pos)
                negative_count.append(neg)
                neutral_count.append(neut)
        return movies, positive_count, negative_count, neutral_count
    except BaseException  as e:
        print e


#Drawing the polarity graph of all movies
def draw_polarity_count_graph(movies,positive_count, negative_count, neutral_count):
    fig, ax = plot.subplots()
    index = np.arange(movies.__len__())
    bar_width = 0.25
    opacity = 0.8

    rects1 = plot.bar(index, negative_count, bar_width,
                     alpha=opacity,
                     color='r',
                     label='Negative')

    rects2 = plot.bar(index + bar_width, positive_count, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Positive')

    rects3 = plot.bar(index + 2*bar_width, neutral_count, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Neutral')
    b = plot.figure(1)
    plot.xlabel('Movies')
    plot.ylabel('Count')
    plot.title('Polarity Count of Movies')
    plot.xticks(index + 1.5 * bar_width, movies ,rotation = 90)
    plot.legend()
    plot.tight_layout()
    l = plot.figure(2)
    plot.plot(range(4),negative_count,color='r',label='Negative')
    plot.plot(range(4),positive_count,color='g', label='Positive')
    plot.plot(range(4), neutral_count, color='b', label='Neutral')
    plot.xlabel('Movies')
    plot.ylabel('Count')
    plot.title('Polarity Count of Movies')
    plot.xticks(index, movies,rotation=90)
    plot.legend()
    plot.tight_layout()
    plot.show()

def main():
    movies, positive_count, negative_count, neutral_count = calculate_polarity_count_of_tweets()
    draw_polarity_count_graph(movies, positive_count, negative_count, neutral_count)

if __name__ == "__main__":
    main()
