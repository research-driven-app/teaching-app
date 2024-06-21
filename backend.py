##General Libraries
import pandas as pd
import numpy as np
import os

## Text Mining
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

##Regex for Data Cleaning
import re


### Detect Language
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


from collections import Counter

import streamlit as st

def detect_language(text):

    # Define a function using langdetect, there is some console printing (we might be able to loose it or integrate it in the final output)

    try:
        return detect(text)
    except LangDetectException as e:
        print(f"Error detecting language: {e}")
        return None
    
def stem_sentence(sentence):

    # Initialize the PorterStemmer
    stemmer = PorterStemmer()

    ##download NLTK data
    nltk.download('punkt')

    ##define the stemming function
    words = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def change_time_columns(df, format_arg='ISO8601'):
    ##Year and Quarter extracted
    df['created_at'] = pd.to_datetime(df['created_at'], format=format_arg)
    df['year'] = df['created_at'].dt.year
    df['quarter']= df['created_at'].dt.quarter
    df['month']= df['created_at'].dt.month
    df['week']= df['created_at'].dt.isocalendar().week

    return df

# Tokenize the text and add absolute counter
def tokenize_and_count(text):
    tokens = nltk.word_tokenize(text.lower())  
    # Convert to lowercase for consistency; this had been done before, so it might be redundant code
    return Counter(tokens)

@st.cache_data
def preprocess(df):

    ## Create a filter for Tweets starting with RT*
    df['retweet'] = df['text'].str.match(r'^RT')

    ## Filter out those that are False on the retweet ,here '~' represents those that evaluate to false
    df = df[~df['retweet']]

    ##Use a function to drop those that are duplicated over three columns
    df = df.drop_duplicates(subset=['text', 'tweet_id'])

    #fill empty spaces (not sure if needed)
    df['text'] = df['text'].fillna('').astype(str)

    # Apply language detection and put it into a new column
    df['language'] = df['text'].apply(detect_language)

    ## For english only filter
    df = df[df['language']== 'en']

    ## Convert to lower case
    df['text'] = df['text'].str.lower() 

    ### Apply stemming

    df['text_stemmed'] = df['text'].apply(stem_sentence)    

    # Apply the function to the 'text' column
    df['term_freq'] = df['text'].apply(tokenize_and_count)

    # Explode the DataFrame so each word gets its own row
    expanded_rows = []
    for _, row in df.iterrows():
        for term, freq in row['term_freq'].items():
            expanded_rows.append([row['tweet_id'], row['text'], row['created_at'], term, freq])


    result_df = pd.DataFrame(expanded_rows, columns=['tweet_id', 'text','created_at', 'term', 'frequency'])

    return result_df


@st.cache_data
def join_and_multiply_data(df, dictionary, old_df):

    joined_df = pd.merge(df, dictionary, on='term', how='inner')

    drives_list = dictionary.columns.tolist()
    drives_list.remove('term')
    
    # Multiply the term frequency with each column of the dictionary
    for col in drives_list:
        joined_df[col] = joined_df['frequency'] * joined_df[col]

    aggregated_df = joined_df.groupby('tweet_id').sum().reset_index()

    old_df = change_time_columns(old_df)

    final_df = pd.merge(old_df, aggregated_df, on='tweet_id', how='inner')

    return final_df

@st.cache_data
def compute_drives(final_df, granularity):
    ## Here we define what columns are grouped by what function, e.g. 'like_count': 'sum' is summing the likes by year and quarter
    grouped_df = final_df.groupby(granularity).agg({
        'tweet_id': 'count',
        'price_pos' : 'sum',
        'price_neg': 'sum',
        'price_pos': 'sum',
        'price_neg': 'sum',
        'squality_pos': 'sum',
        'squality_neg': 'sum',
        'goodsq_pos': 'sum',
        'goodsq_neg' : 'sum',
        'cool_pos': 'sum',
        'cool_neg': 'sum',
        'exciting_pos': 'sum',
        'exciting_neg': 'sum',
        'innov_pos': 'sum',
        'innov_neg': 'sum',
        'socresp_pos': 'sum',
        'socresp_neg': 'sum',
        'comm_pos': 'sum',
        'comm_neg': 'sum',
        'friendly_pos': 'sum',
        'friendly_neg': 'sum',
        'personalrel_pos': 'sum',
        'personalrel_neg': 'sum',
        'trust_pos': 'sum',
        'trust_neg': 'sum'
    }).reset_index()

    grouped_df = grouped_df.rename(columns={'tweet_id': 'total_tweets'})


    ##Value Driver
    grouped_df['Price_net'] = grouped_df['price_pos'] - grouped_df['price_neg']
    grouped_df['Gquality_net'] = grouped_df['squality_pos'] - grouped_df['squality_neg']
    grouped_df['Goodsq_net'] = grouped_df['goodsq_pos'] - grouped_df['goodsq_neg']
    ## To establish Value Driver sum the three subdrivers and take the average
    grouped_df['Value_Driver'] = (grouped_df['Price_net'] + grouped_df['Gquality_net'] + grouped_df['Goodsq_net'])/3


    ##Brand Driver
    grouped_df['Cool_net'] = grouped_df['cool_pos'] - grouped_df['cool_neg']
    grouped_df['Innovative_net'] = grouped_df['innov_pos'] - grouped_df['innov_neg']
    grouped_df['Exciting_net'] = grouped_df['exciting_pos'] - grouped_df['exciting_neg']
    grouped_df['SocResp_net'] = grouped_df['socresp_pos'] - grouped_df['socresp_neg']
    ## To establish Brand Driver sum the four subdrivers and take the average
    grouped_df['Brand_Driver'] = (grouped_df['Cool_net'] + grouped_df['Innovative_net'] + grouped_df['Exciting_net']+ grouped_df['SocResp_net'])/4

    ##Relationship Driver
    grouped_df['Community_net'] = grouped_df['comm_pos'] - grouped_df['comm_neg']
    grouped_df['Friendly_net'] = grouped_df['friendly_pos'] - grouped_df['friendly_neg']
    grouped_df['PersonalRel_net'] = grouped_df['personalrel_pos'] - grouped_df['personalrel_neg']
    grouped_df['Trustworthy_net'] = grouped_df['trust_pos'] - grouped_df['trust_neg']
    ## To establish Relationship Driver sum the four subdrivers and take the average
    grouped_df['Relationship_Driver'] = (grouped_df['Community_net'] + grouped_df['Friendly_net'] + grouped_df['PersonalRel_net']+ grouped_df['Trustworthy_net'])/4

    grouped_df['Brand Reputation'] = grouped_df['Value_Driver'] + grouped_df['Brand_Driver'] + grouped_df['Relationship_Driver']

    # Define Columns to normalize

    driver_columns = ['Price_net', 'Gquality_net', 'Goodsq_net', 'Value_Driver',
                    'Cool_net', 'Innovative_net', 'Exciting_net', 'SocResp_net', 'Brand_Driver',
                    'Community_net', 'Friendly_net', 'PersonalRel_net', 'Trustworthy_net', 'Relationship_Driver']

    # Perform z-score normalization: 1) first means, then standard deviation 2) substract the mean from the value and divide by standard deviation
    for column in driver_columns:
        mean = grouped_df[column].mean()
        std = grouped_df[column].std()
        grouped_df[column] = (grouped_df[column] - mean) / std

    driver_columns = driver_columns + ['Brand Reputation']

    return grouped_df[driver_columns]