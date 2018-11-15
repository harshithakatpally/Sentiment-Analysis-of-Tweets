from pandas import json
import re
import tweepy
import pickle
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener


#Class for fetching twitter data
class TwitterClient(object):
    def __init__(self):
        consumer_key = 'xxxxxxxxxxxxxxxxx'
        consumer_secret = 'xxxxxxxxxxxxxxxxx'
        access_token = 'xxxxxxxxxxxxxxxxx'
        access_secret = 'xxxxxxxxxxxxxxxxx'

        #Attempt authentication to get access to twitter data and create twitter API object
        try:
            self.auth = OAuthHandler(consumer_key, consumer_secret)
            self.auth.set_access_token(access_token, access_secret)
            self.api = tweepy.API(self.auth)
        except:
            print ("Error: Authentication failed")

    #Function to clean tweet text by removing links, special characters using simple regex statements.
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | \
        (\w+:\/\/\S+)", " ", tweet).split())

    #Function to fetch tweets from twitter and parse them
    def get_tweets(self, search, max_tweets = 20):
        fetched_tweets = []
        try:
            tweets = self.api.search(q = search, count = max_tweets)

            for tweet in tweets:
                if tweet.retweet_count > 0:
                    if tweet.text not in fetched_tweets:
                        fetched_tweets.append(tweet.text)
                else:
                    fetched_tweets.append(tweet.text)
            return fetched_tweets
        except tweepy.TweepError as e:
             print("Error : " + str(e))


def main():
    api = TwitterClient()
    #Change the search keyword to extract tweets related to different movies
    search_keyword = "Johnny English"
    tweets = api.get_tweets(search = search_keyword, max_tweets = 20)
    #Save the tweets to a text file
    with open(search_keyword + ".txt", "w") as f:
        for tweet in tweets:
            tweet = tweet.encode('utf-8')
            f.write(str(tweet) +"\n")

if __name__ == "__main__":
    main()
