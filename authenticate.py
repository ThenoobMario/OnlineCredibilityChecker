import tweepy as tw
from os import environ

def authenticate():
    # Entering the credentials for authentication
    apiKey = environ['apiKey']
    apiSecret = environ['apiSecret']
    apiAccess = environ['apiAccess']
    apiAccessSecret = environ['apiAccessSecret']

    # Authenticating
    auth = tw.AppAuthHandler(apiKey, apiSecret)
    api = tw.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

    return api
