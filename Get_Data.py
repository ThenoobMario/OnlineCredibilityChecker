import pandas as pd
import tweepy as tw
from authenticate import authenticate

def getUserData(username, n):
    # Running the authentication
    api = authenticate()

    # Fetching data
    # Here the number inside items can be modified to retrieve more tweets
    userData = tw.Cursor(api.user_timeline, id = username, include_rts = False).items(n)

    # Creating a dataframe of the required data
    return createDF(userData)

def createDF(data):
    # Creating empty Lists to store data
    tweetList = []
    hashtagList = []
    timeList = []
    likeList = []
    rtList = []

    # Parsing through the JSON objects to collect relevant
    for tweet in data:
        tweetList.append(tweet.text)
        hashtagList.append(tweet.entities.get('hashtags'))
        timeList.append(tweet.created_at)
        likeList.append(tweet.favorite_count)
        rtList.append(tweet.retweet_count)

    # Making a DataFrame
    twtDf = pd.DataFrame({'Created At': timeList,
                        'tweets': tweetList,
                        'hashtags': hashtagList,
                        'likes': likeList,
                        'retweets': rtList})

    # Checking if the data fetched is correct
    # print(twtDf.head())

    return twtDf
