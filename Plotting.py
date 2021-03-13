import plotly.express as px
from collections import Counter
from Preprocess import *
import numpy as np
import pandas as pd
from wordcloud import WordCloud

# Counts the number of links shared
def numLinksShared(listOfLinks):
    h = []

    for x in listOfLinks:
        h.extend(x)

    h = set(h)

    return len(h)


def createSortedDict(counts, num):
    sorted_tuples = sorted(counts.items(), key = lambda item: item[1], reverse = True)
    sorted_dict = {}

    for x, y in sorted_tuples[:num]:
        sorted_dict[x] = y

    return sorted_dict

# To find vocab size and frequently used words
def numWords(listOfStrings, num):
    h = []

    for x in listOfStrings:
        h.extend(x.split())

    counts = Counter(h)

    sorted_dict = createSortedDict(counts, num)

    fig = px.bar(x = sorted_dict.keys(), y = sorted_dict.values(),
                labels = {'x': 'Words', 'y': 'Number of times used'},
                title = 'Word Usage')

    return fig, len(set(h))


def avgUniqueWords(uniqueWordList):
    avg = np.mean(uniqueWordList)
    std = np.std(uniqueWordList)

    return avg, std

# Returns bar plot of number of emojis, unique emojis used and emoji frequency
def numEmojis(listOfEmojis, num):
    h = []

    for x in listOfEmojis:
        h.extend(x)

    if len(h) == 0:
        return px.bar(title = 'No Emojis used'), 0, 0

    counts = Counter(h)

    sorted_dict = createSortedDict(counts, num)

    fig = px.bar(x = sorted_dict.keys(), y = sorted_dict.values(),
                labels = {'x': 'Emojis', 'y': 'Number of times used'},
                title = 'Emoji Usage')

    return fig, len(set(h)), len(h)


# Returns bar plot of number of hashtags, unique hashtags used and hastag frequency
def numHashtags(listOfHashtags, num):
    h = []

    for x in listOfHashtags:
        h.extend(x)

    if len(h) == 0:
        return px.bar(title = 'No Hastags used'), 0, 0

    counts = Counter(h)

    sorted_dict = createSortedDict(counts, num)

    df_sorted = pd.DataFrame({'Keys':sorted_dict.keys(),
                            'Values':sorted_dict.values()})

    fig = px.bar(x = df_sorted['Keys'], y = df_sorted['Values'],
                labels = {'x': 'Hashtags','y': 'Number of times used'},
                title = 'Hashtag Usage')

    return fig, len(set(h)), len(h)

# df.replace()
def tweetClusterClassfier(cleanedTweets, likes, retweet):
    # Predicting the cluster of all tweets
    preds = predictCluster(cleanedTweets)
    df = pd.DataFrame({'Tweet': cleanedTweets, 'Prediction': preds,
                        'Likes': likes, 'Retweets': retweet})

    df['Prediction'].replace({0: 'Confident and Hopeful',
                                1: 'Complacent and Rational',
                                2: 'Stressful and Worrisome',
                                3: 'Excited and Humble',
                                4: 'Iterative and Reminiscing',
                                5: 'Nostalgic and Needy',
                                6: 'Driven and Tired',
                                7: 'Spontaneous and Fun-Loving',
                                8: 'People-Pleasing and Sensitive',
                                9: 'Expressive and Honest'}, inplace = True)

    count = pd.value_counts(df['Prediction'])
     # Finding most frequent tweet type
    mostFreq = count.index[np.argmax(count)]
    likeCount = df.groupby('Prediction')[['Likes', 'Retweets']].agg(np.mean).reset_index()  # average likes based on tweet cluster
    # Correlation between Cluster and the average likes
    likeFig = px.bar(x = likeCount['Prediction'], y = likeCount['Likes'],
                    title = 'Average likes for each cluster',
                    labels = {'x': 'Clusters', 'y': 'Average Number of Likes'})
    # Correlation between Cluster and the average Retweets
    retweetFig = px.bar(x = likeCount['Prediction'], y = likeCount['Retweets'],
                        title = 'Average retweet for each cluster',
                        labels = {'x': 'Clusters', 'y': 'Average Number of Retweets'})

    df = df[df['Prediction'] == mostFreq]

    # 15 most common words in most common cluster
    fig, num = numWords(df['Tweet'], 15)

    return fig, num, mostFreq, likeFig, retweetFig


# df=generate_tweet_data('GyaneshShah',500)
# print(df.shape)
# print(df.head()[['cleaned_hashtags','emojisUsed']])
# fig,num,cluster,likes_fig,retweet_fig=tweetClusterClassfier(df['Cleaned_tweets'],df['likes'],df['retweets'])
# fig.show()
# print(num)
# print(cluster)
# likes_fig.show()
# retweet_fig.show()
# fig,num,totalUsed=numEmojis(df['emojisUsed'], 50)
# fig.show()
# print('Number of unique emojis: ',num)
# print('Total emojis used : ', totalUsed)
# fig,num,totalUsed=numHashtags(df['cleaned_hashtags'], 50)
# fig.show()
# print('Number of unique hastags ',num)
# print('Total hashtags used : ',totalUsed)
# fig,num=numWords(df['Cleaned_tweets'],100)
# fig.show()
# print('Number of unique words used ',num)
#
# print(numLinksShared(df['links']))
# print('The average tweet length is : ',np.mean(df['tweetLength']))
# print('The average words used in a tweet : ', np.median(df['numberOfUniqueWords']))
