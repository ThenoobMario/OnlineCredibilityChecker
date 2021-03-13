from nltk.corpus import stopwords
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA,TruncatedSVD,LatentDirichletAllocation
from sklearn.naive_bayes import GaussianNB
import joblib
import plotly.express as px
import seaborn as sns

# Read the data and clean it
data = pd.read_csv('./data/tweets_sentiment.csv', encoding = 'ISO-8859-1', usecols=[0, 1])
data['SentimentText'] = data['SentimentText'].str.strip()
data['SentimentText'] = data['SentimentText'].str.lower()

def clean_data(x):
    x = x.lower()
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    x = re.sub(regex,'',x)   # replaces the url
    x = x.strip()
    cleaned_tweet_words=[]
    stop = stopwords.words('english')    # generates a list of stop words
    #x=demoji.replace(x,'')         # replaces the emojis in the string with their description
    x = re.sub('[:!,?@._]',' ',x)# removing puncutations
    x = re.sub('[^A-Za-z]',' ',x)
    x = x.split(" ")

    for word in x:
        if word not in stop and not word.strip().startswith('#'):
            # if word is not stop word and a hashtag then append
            cleaned_tweet_words.append(word.strip())
    return ' '.join(cleaned_tweet_words)           # returns string of cleaned words


data['tweet'] = data['SentimentText'].apply(lambda x:clean_data(x))
data.head()

vectorizer = TfidfVectorizer(max_features = 50000)
X = vectorizer.fit_transform(data['tweet'])

kmeans = KMeans(n_clusters = 10)
kmeans.fit(X)



centroid = kmeans.cluster_centers_.argsort()[:, ::-1]
terms = vectorizer.get_feature_names()

for i in range(10):
    print()
    print('Cluster: ', i)

    for ind in centroid[i, :10]:
        print(terms[ind], end = ',')


pipeline = Pipeline([('Vectorizer', TfidfVectorizer()), ('kmeans', KMeans(n_clusters = 5))])
pipeline.fit(data['tweet'])

filename = 'model.sav'
joblib.dump(pipeline, filename)
