##General Libraries
import pandas as pd
import numpy as np
import os

## Text Mining
import nltk

##download NLTK data
nltk.download('punkt')
nltk.data.path.append('nltk_data')
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


### Detect Language
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException


from collections import Counter

import streamlit as st

def check_columns_for_neg_suffix(df_check, string):
    # Iterate through each column name in the DataFrame
    for column in df_check.columns:
        # Check if '_neg' is a substring of the column name
        if string in column:
            # Return False if '_neg' is found
            return False
    # Return True if '_neg' is not found in any column name
    return True

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

    ##define the stemming function
    words = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_words)

def change_time_columns(df, format_arg='ISO8601'):
    # Ensure we are working with a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()
    
    # Convert 'created_at' to datetime
    df['created_at'] = pd.to_datetime(df['created_at'], format=format_arg)
    
    # Extract year, quarter, month, and week
    df.loc[:, 'year'] = df['created_at'].dt.year
    df.loc[:, 'quarter'] = df['created_at'].dt.quarter
    df.loc[:, 'month'] = df['created_at'].dt.month
    df.loc[:, 'week'] = df['created_at'].dt.isocalendar().week

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


    result_df = pd.DataFrame(expanded_rows, columns=['tweet_id', 'text', 'created_at', 'term', 'frequency'])

    return result_df


@st.cache_data
def join_and_multiply_data(df_join, dictionary, old_df, timestamp_format_macro='ISO8601', extra_dict=None):

    if extra_dict is not None:
        dictionary = pd.merge(dictionary, extra_dict, on='term', how='inner')

    joined_df = pd.merge(df_join, dictionary, on='term', how='inner')

    drives_list = dictionary.columns.tolist()
    drives_list.remove('term')
    
    # Multiply the term frequency with each column of the dictionary
    for col in drives_list:
        joined_df[col] = joined_df['frequency'] * joined_df[col]

    joined_df = joined_df[drives_list + ['tweet_id']]
    aggregated_df = joined_df.groupby('tweet_id').sum().reset_index()


    old_df = change_time_columns(old_df,  format_arg = timestamp_format_macro)

    merged_df = pd.merge(old_df, aggregated_df, on='tweet_id', how='inner')

    return merged_df

def prepare_aggregate_dict(list_of_drives):
    aggregate_dict = {}
    aggregate_dict['tweet_id'] = 'count'
    for drive in list_of_drives:
        aggregate_dict[drive+"_pos"] = 'sum'
        aggregate_dict[drive+"_neg"] = 'sum'
    return aggregate_dict


@st.cache_data
def compute_drives(final_df, granularity, extra_driver=None):
    dict_columns_ids = ["price", "squality", "goodsq", "cool", "exciting", "innov", "socresp", "comm", "friendly", "personalrel", "trust"]
    if extra_driver is not None:
        dict_columns_ids.append(extra_driver)

    dict_to_aggregate = prepare_aggregate_dict(dict_columns_ids)

    dict_to_aggregate["created_at"] = 'first'

    grouped_df = final_df.groupby(granularity).agg(dict_to_aggregate).reset_index()

    grouped_df = grouped_df.rename(columns={'tweet_id': 'total_tweets'})

    dict_of_drivers = {}
    dict_of_drivers["Value"] = ["price", "squality", "goodsq"]
    dict_of_drivers["Brand"] = ["cool", "exciting", "innov", "socresp"]
    dict_of_drivers["Relationship"] = ["comm", "friendly", "personalrel", "trust"]

    if extra_driver is not None:
        dict_of_drivers[extra_driver] = [extra_driver]

    driver_columns = []

    for driver in dict_of_drivers:

        for value in dict_of_drivers[driver]:

            new_column = value+"_net"
            grouped_df[new_column] = grouped_df[value+"_pos"] - grouped_df[value+"_neg"]
            driver_columns.append(new_column)

        computed_nets = [value+"_net" for value in dict_of_drivers[driver]]
        new_column_driver = driver+"_Driver"
        grouped_df[new_column_driver] = grouped_df[computed_nets].mean(axis=1)
        driver_columns.append(new_column_driver)        

    grouped_df['Brand Reputation'] = grouped_df['Value_Driver'] + grouped_df['Brand_Driver'] + grouped_df['Relationship_Driver']
    if extra_driver is not None:
        grouped_df['Brand Reputation'] = grouped_df['Brand Reputation'] + grouped_df[extra_driver+"_Driver"]

    # Perform z-score normalization: 1) first means, then standard deviation 2) substract the mean from the value and divide by standard deviation
    for column in driver_columns:
        mean = grouped_df[column].mean()
        std = grouped_df[column].std()
        grouped_df[column] = (grouped_df[column] - mean) / std

    driver_columns = driver_columns + ['Brand Reputation']
    
    grouped_df.sort_values(by='created_at', inplace=True)
    grouped_df.set_index("created_at", inplace=True)
    grouped_df = grouped_df[driver_columns]
    return grouped_df

