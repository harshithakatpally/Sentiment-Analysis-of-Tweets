from pandas import json
import tweepy
from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener

#Keys generating after registering the app on twitter
consumer_key = 'k6gI0TbRp7NKHQGOlArOeWCGe'
consumer_secret = 'xy5yB5nHx8lBrl56j7ZnoNmwW5mpQB3dl8pBFEffkCkpl9byBq'
access_token = '3267228391-Xyuj9YQxRlChW27FPqSBt4pCnuPxkJ89GqPNP3Z'
access_secret = 'tEmifh21feNNtwtN5Kh63DNa4GofJryAUeTDlgjncO4j1'

#Creating OAuthHandler object to set access, secret and token and
#twitter API object to fetch tweets
auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)

def process_or_store(tweet):
    print(json.dumps(tweet))

#Listens to data coming from twitter and copies it into a jason file.
class MyListener(StreamListener):
    def on_data(self, data):
        try:
            with open('naamshabana.json', 'a') as f:
                f.write(data)
                return True
        except BaseException as e:
            print("Error on_data: %s "%str(e))
            return True

    def on_error(self, status):
        print(status)
        return True

twitter_stream = Stream(auth, MyListener())
#Use the key word specific to the type of data required
twitter_stream.filter(track=['naam shabana'])
