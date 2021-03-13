import demoji
import joblib
from Get_Data import getUserData
import re
from nltk.corpus import stopwords


# Declaring a global regex string
regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"

def predictCluster(list_of_tweets):
    pipeline = joblib.load('model.sav')
    pred = pipeline.fit_predict(list_of_tweets)

    return pred


def getLinks(string):
    url = re.findall(regex, string)

    if len(url)!=0:
        url=[x for x in url[0] if len(x) != 0]
        return url
    else:
        return []


def cleanData(x):
    x = x.lower()
    cleaned_tweet_words = []

    # Replace the url
    x = re.sub(regex, '', x)
    x = x.strip()
    # removing usernames
    x = re.sub('@[^\s]+', '', x)
    # Generate a list of stop words
    stop = stopwords.words('english')
    # Replace the emojis in the string with their description
    x = demoji.replace(x,'')
    # Removing puncutations
    x = re.sub('[:!,?@]', ' ', x)
    x = x.split(" ")

    for word in x:
        if word not in stop and not word.strip().startswith('#'):
            # If word is neither a stop word nor a hashtag then append
            cleaned_tweet_words.append(word.strip())

    return ' '.join(cleaned_tweet_words)

# Finding total mentions from the extracted tweets
def findMentions(x):
    x = x.split(' ')
    mentions = []
    for word in x:
        if word.strip().startswith('@'):
            mentions.append(word)
    return mentions

# Defining all the getter and count functions
def getHashtags(x):
    hashtags = []

    for y in x:
        hashtags.append(y['text'])

    return hashtags

def getEmojis(x):
    h = demoji.findall(x)

    return h.keys()

def countQuestionmarks(x):
    return x.count('?')

def countExclamation(x):
    return x.count('!')


def uniqueWords(x):
    x = re.sub(regex, '', x)  # replaces the url
    x = x.strip()
    x = set(x.split())

    return len(x)


def generateTweetData(username, numTweets):
    df = getUserData(username, numTweets)

    df['Cleaned_tweets'] = df['tweets'].apply(lambda x:cleanData(x))
    df['links'] = df['tweets'].apply(lambda x:getLinks(x))
    df['cleaned_hashtags'] = df['hashtags'].apply(lambda x:getHashtags(x))
    df['emojisUsed'] = df['tweets'].apply(lambda x:getEmojis(x))
    df['QuestionMarksInTweet'] = df['tweets'].apply(lambda x:countQuestionmarks(x))
    df['ExclamationMarksInTweet'] = df['tweets'].apply(lambda x:countExclamation(x))
    df['Mentions'] = df['tweets'].apply(lambda x: findMentions(x))
    df['tweetLength'] = df['tweets'].apply(lambda x:len(x))
    df['numberOfUniqueWords'] = df['Cleaned_tweets'].apply(lambda x:uniqueWords(x))

    return df
