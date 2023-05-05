import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
from textblob import TextBlob
import cleantext
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

DATA_URL = (
    "Tweets.csv"
)

st.title("Twitter Sentiment Analysis")
st.sidebar.title("Sentiment Analysis of Tweets")
st.markdown("This application is a Streamlit dashboard used "
            "to analyze sentiments of tweets and to provide single tweet analysis")
st.sidebar.markdown("This application is a Streamlit dashboard used "
                    "to analyze sentiments of tweets ")

@st.cache_data
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

def convert_to_df(sentiment):
	sentiment_dict = {'polarity':sentiment.polarity,'subjectivity':sentiment.subjectivity}
	sentiment_df = pd.DataFrame(sentiment_dict.items(),columns=['metric','value'])
	return sentiment_df

data = load_data()

st.write("Please enter the choice : 1(For Single Tweet analysis and  for Tweet Cleaning) or 2(Sentiment Analysis of US airlines Data)")
k1 = st.text_input("Enter choice(Default is Choice Number 2)")

if k1 == '1':
    st.subheader('Sentiment Analysis Of Single Tweet(Tweet Cleaning can be done here too)')
    with st.expander('Analyze Text'):
        text = st.text_input('Text here: ')

        sentiment = TextBlob(text).sentiment
        st.write(sentiment)

        result_df = convert_to_df(sentiment)
        st.dataframe(result_df)

        c = alt.Chart(result_df).mark_bar().encode(
					x='metric',
					y='value',
					color='metric')
        st.altair_chart(c,use_container_width=True)    

        pre = st.text_input('Clean Text: ')
        if pre:
            st.write(cleantext.clean(pre, clean_all= False, extra_spaces=True ,
                                    stopwords=True ,lowercase=True ,numbers=True , punct=True))        
    

else :
    with st.expander('sidegigs'):


        st.sidebar.subheader("Show random tweet")
        random_tweet = st.sidebar.radio('Sentiment', ('positive', 'neutral', 'negative'))
        st.sidebar.markdown(data.query("airline_sentiment == @random_tweet")[["text"]].sample(n=1).iat[0, 0])

        st.sidebar.markdown("### Number of tweets by sentiment")
        select = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='1')
        sentiment_count = data['airline_sentiment'].value_counts()
        sentiment_count = pd.DataFrame({'Sentiment':sentiment_count.index, 'Tweets':sentiment_count.values})
        if not st.sidebar.checkbox("Hide", True):
            st.markdown("### Number of tweets by sentiment")
            if select == 'Bar plot':
                fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color='Tweets', height=500)
                st.plotly_chart(fig)
            else:
                fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
                st.plotly_chart(fig)

        st.sidebar.subheader("When and where are users tweeting from?")
        hour = st.sidebar.slider("Hour to look at", 0, 23)
        modified_data = data[data['tweet_created'].dt.hour == hour]
        if not st.sidebar.checkbox("Close", True, key = 23):
            st.markdown("### Tweet locations based on time of day")
            st.markdown("%i tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour + 1) % 24))
            st.map(modified_data)
            if st.sidebar.checkbox("Show raw data", False):
                st.write(modified_data)


        st.sidebar.subheader("Total number of tweets for each airline")
        each_airline = st.sidebar.selectbox('Visualization type', ['Bar plot', 'Pie chart'], key='2')
        airline_sentiment_count = data.groupby('airline')['airline_sentiment'].count().sort_values(ascending=False)
        airline_sentiment_count = pd.DataFrame({'Airline':airline_sentiment_count.index, 'Tweets':airline_sentiment_count.values.flatten()})
        if not st.sidebar.checkbox("Close", True, key = 76):
            if each_airline == 'Bar plot':
                st.subheader("Total number of tweets for each airline")
                fig_1 = px.bar(airline_sentiment_count, x='Airline', y='Tweets', color='Tweets', height=500)
                st.plotly_chart(fig_1)
            if each_airline == 'Pie chart':
                st.subheader("Total number of tweets for each airline")
                fig_2 = px.pie(airline_sentiment_count, values='Tweets', names='Airline')
                st.plotly_chart(fig_2)


        @st.cache_data
        def plot_sentiment(airline):
            df = data[data['airline']==airline]
            count = df['airline_sentiment'].value_counts()
            count = pd.DataFrame({'Sentiment':count.index, 'Tweets':count.values.flatten()})
            return count


        st.sidebar.subheader("Breakdown airline by sentiment")
        choice = st.sidebar.multiselect('Pick airlines', ('US Airways','United','American','Southwest','Delta','Virgin America'))
        if len(choice) > 0:
            st.subheader("Breakdown airline by sentiment")
            breakdown_type = st.sidebar.selectbox('Visualization type', ['Pie chart', 'Bar plot', ], key='3')
            fig_3 = make_subplots(rows=1, cols=len(choice), subplot_titles=choice)
            if breakdown_type == 'Bar plot':
                for i in range(1):
                    for j in range(len(choice)):
                        fig_3.add_trace(
                            go.Bar(x=plot_sentiment(choice[j]).Sentiment, y=plot_sentiment(choice[j]).Tweets, showlegend=False),
                            row=i+1, col=j+1
                        )
                fig_3.update_layout(height=600, width=800)
                st.plotly_chart(fig_3)
            else:
                fig_3 = make_subplots(rows=1, cols=len(choice), specs=[[{'type':'domain'}]*len(choice)], subplot_titles=choice)
                for i in range(1):
                    for j in range(len(choice)):
                        fig_3.add_trace(
                            go.Pie(labels=plot_sentiment(choice[j]).Sentiment, values=plot_sentiment(choice[j]).Tweets, showlegend=True),
                            i+1, j+1
                        )
                fig_3.update_layout(height=600, width=800)
                st.plotly_chart(fig_3)


        st.sidebar.header("Word Cloud")
        word_sentiment = st.sidebar.radio('Display word cloud for what sentiment?', ('positive', 'neutral', 'negative'))
        if not st.sidebar.checkbox("Close", True, key = 6):
            st.subheader('Word cloud for %s sentiment' % (word_sentiment))
            df = data[data['airline_sentiment']==word_sentiment]
            words = ' '.join(df['text'])
            processed_words = ' '.join([word for word in words.split() if 'http' not in word and not word.startswith('@') and word != 'RT'])
            wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', width=800, height=640).generate(processed_words)
            plt.imshow(wordcloud)
            plt.xticks([])
            plt.yticks([])
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

