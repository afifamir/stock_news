import streamlit as st
import sys
import subprocess
import requests
import nltk
import yfinance as yf
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
from nixtlats import TimeGPT
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from datetime import datetime 
import string

def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenize text
    tokens = word_tokenize(text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Rejoin tokens into a string
    text = ' '.join(tokens)

    return text

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

def get_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    scores = analyzer.polarity_scores(text)
    sentiment = scores['compound']
    return sentiment

def get_stock_news_sentiment(api_key, stock):
    # Make request to NewsAPI
    url = f'https://newsapi.org/v2/everything?q={stock}&apiKey={api_key}'
    response = requests.get(url)
    data = response.json()
    sentiments = []
    for article in data['articles']:
        # Preprocess text
        text = article['title'] + ' ' + article['description']
        text = preprocess_text(text)
        
        # Perform sentiment analysis
        sent = get_sentiment(text)
        sentiments.append(sent)
    # Calculate average sentiment
    if len(sentiments) > 0:
        avg_sentiment = sum(sentiments) / len(sentiments)
        avg_sentiment = "{:.2f}".format(avg_sentiment)
    else:
        avg_sentiment = 0

    return avg_sentiment

def get_news(api_key, stock):
    url = f'https://newsapi.org/v2/everything?q={stock}&apiKey={api_key}'
    response = requests.get(url)
    data = response.json()
    title = []
    description = []
    for article in data['articles']:
        # Preprocess text
        df = pd.DataFrame()
        title1 = article['title']
        title.append(title1)
        description1 = article['description']
        description.append(description1)
        df['Title'] = title
        df['Description'] = description
        return df


api_key = 'c3dbf0fac1e64daa84c80b0b87828c6c'



def main():
    menu = ["Home", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Home":
        st.subheader("Sentiment Analysis for Your Stockpick!")
        st.markdown("You choose your stock, we will go through the news and simplify the sentiment through number.")

        with st.form(key = "stock_form"):
            raw_text = st.text_input("Type the stock's ticker here")
            submit_text = st.form_submit_button(label = 'Analyze now!')

        if submit_text:
            col1, col2 = st.columns(2)

            prediction = get_stock_news_sentiment('c3dbf0fac1e64daa84c80b0b87828c6c', {raw_text})
            stock = yf.Ticker(raw_text)
            start_date = datetime(2020, 1, 1) 
            end_date = datetime.today()

            data = yf.download(raw_text.upper(), start = start_date, end = end_date)['Adj Close']
            st.success("Chart")
            st.line_chart(data)

            st.success('Recent Stock Price')
            price = stock.history(period="12d")[['Open','Close']]
            st.dataframe(price)

            #timegpt = TimeGPT(token='G7C7uAUNmc1EGtjw8jLKWbzexQokLJFuZ43dNKGkZbPEi4rvJruJGZF534ya3N3gOH5MbJJdXt8mHCMP49Zon0sSbuzmKs77DJdjDfxN9qmkNALB2pl0jXOSHL2yVy9gfOQg3O5cXVEqbrb6w9taCJ1DuLkugnqp3LCSGCu6SLD8aRRocEpreSLMcd2Sx272PJi7jBimp8n9KRsjUSfS21MYTG1OCXyhFLEcNqbjUPeBVZkLzugNxBZKhbAZWHPI')

            #timegpt_fcst_df = timegpt.forecast(df=price, h = 12)
            #, h=12, freq='MS', time_col='timestamp', target_col='Close'
            #st.dataframe(timegpt_fcst_df)


            with col1:
                st.success("Ticker")
                st.write(raw_text)

                #st.success("Prediction")
                st.metric(label = "Sentiment", value = prediction)
            
            with col2:
                news = get_news('c3dbf0fac1e64daa84c80b0b87828c6c', {raw_text})
                st.success("Recent news regarding this ticker")
                st.dataframe(
                    news,
                    hide_index=True,
                )
                

    else:
        st.subheader("About")

if __name__ == '__main__':
    main()