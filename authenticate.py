import tweepy as tw

def authenticate():
    # Entering the credentials for authentication
    apiKey = 'API KEY'
    apiSecret = 'SECRET API KEY'
    apiAccess = 'ACCESS KEY'
    apiAccessSecret = 'ACCESS SECRET KEY'

    # Authenticating
    auth = tw.AppAuthHandler(apiKey, apiSecret)
    api = tw.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

    return api
