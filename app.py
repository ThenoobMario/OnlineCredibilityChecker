import streamlit as st
from Preprocess import generateTweetData
from Plotting import tweetClusterClassfier, numHashtags

# Just making sure we are not bothered by File Encoding warnings
st.set_option('deprecation.showfileUploaderEncoding', False)

def main():
    # Title of our webpage
    st.title("Online Credibility Checker")
    st.subheader("By ThenoobMario & NidhayPancholi")

    username = st.text_input("Enter the Twitter username you want to search for (@username):")
    numTweets = st.slider("Select number of Tweets to Extract", 50, 1000)

    if st.button('Predict'):
        df = generateTweetData(username, numTweets)

        fig, uniqueWords, cluster, likesFig, retweetFig = tweetClusterClassfier(df['Cleaned_tweets'], df['likes'], df['retweets'])

        st.markdown("---")
        st.header("The Result")
        st.write("The **Online Persona** of the person is ", cluster, ".")

        st.markdown("### Here are some plots for greater Insights:")
        st.write("- Number of **unique words** used by the person:", uniqueWords)

        st.markdown("- The person uses the following **words** frequently:")
        st.plotly_chart(fig)

        st.markdown("- Here is a graph showing the Correlation between **Average Likes** and the **Clusters**:")
        st.plotly_chart(likesFig)

        st.markdown("- Here is a graph showing the Correlation between **Average Retweets** and the **Clusters**:")
        st.plotly_chart(retweetFig)

        st.markdown("- The person uses the following **Hashtags** frequently:")
        hashFig = numHashtags(df['cleaned_hashtags'], 15)
        st.plotly_chart(hashFig)

    

if __name__ == "__main__":
    main()
