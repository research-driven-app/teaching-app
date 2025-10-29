# ==============================================================
# BACKEND PAGE - backend.py
# --------------------------------------------------------------
# Handles text mining, data preprocessing, joining, aggregation,
# per-tweet scoring, and exporting in UTF-8 / Looker-ready format
# ==============================================================

import os
import io
import pandas as pd
import numpy as np
import nltk
from collections import Counter
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st

# --------------------------------------------------------------
# Setup: NLTK data
# --------------------------------------------------------------
nltk.download("punkt", quiet=True)
nltk.data.path.append("nltk_data")

# --------------------------------------------------------------
# Utility Functions
# --------------------------------------------------------------

def check_columns_for_neg_suffix(df_check, string):
    """
    Return False if the substring exists in any column name
    (used for checking *_neg columns in dictionaries).
    """
    for column in df_check.columns:
        if string in column:
            return False
    return True


def detect_language(text):
    """
    Detect language using langdetect; return None if detection fails.
    """
    try:
        return detect(text)
    except LangDetectException:
        return None


def stem_sentence(sentence):
    """
    Apply Porter stemming to a sentence.
    """
    stemmer = PorterStemmer()
    words = nltk.word_tokenize(sentence)
    stemmed_words = [stemmer.stem(w) for w in words]
    return " ".join(stemmed_words)


def change_time_columns(df, format_arg=None, dayfirst=True, errors="coerce"):
    """
    Convert 'created_at' column to datetime and add temporal features:
    year, quarter, month, ISO week.

    This ALSO keeps created_at as datetime64[ns] (no timezone).
    """
    df = df.copy()
    try:
        df["created_at"] = pd.to_datetime(
            df["created_at"],
            format=format_arg,
            dayfirst=dayfirst,
            errors=errors,
        )

        # Strip timezone if any
        try:
            df["created_at"] = df["created_at"].dt.tz_localize(None)
        except Exception:
            pass

        valid_dates = df["created_at"].notna()
        df.loc[valid_dates, "year"] = df.loc[valid_dates, "created_at"].dt.year
        df.loc[valid_dates, "quarter"] = df.loc[valid_dates, "created_at"].dt.quarter
        df.loc[valid_dates, "month"] = df.loc[valid_dates, "created_at"].dt.month
        df.loc[valid_dates, "week"] = df.loc[valid_dates, "created_at"].dt.isocalendar().week

    except Exception as e:
        print(f"Error processing datetime column: {e}")

    return df


def tokenize_and_count(text):
    """
    Tokenize a text row into lowercase word tokens
    and return a Counter of term frequencies.
    """
    tokens = nltk.word_tokenize(str(text).lower())
    return Counter(tokens)


# --------------------------------------------------------------
# Robust helpers for missing/dirty 'retweet' columns
# --------------------------------------------------------------

def _coerce_retweet(series: pd.Series) -> pd.Series:
    """
    Convert any possible 'retweet' column into a clean boolean Series.
    Handles:
    - True/False
    - 1/0, 1.0/0.0
    - 'true'/'false', 'yes'/'no', '1'/'0'
    - NaN
    """
    if pd.api.types.is_bool_dtype(series):
        out = series.astype(bool)
    elif pd.api.types.is_integer_dtype(series) or pd.api.types.is_float_dtype(series):
        out = series.fillna(0).astype(int).astype(bool)
    else:
        out = (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
                "yes": True,
                "no": False,
            })
            .fillna(False)
        )
    return out.astype(bool)


def _filter_out_retweets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure a 'retweet' column exists, clean/coerce it,
    and drop rows that are retweets. If we can't detect it,
    we assume they're all original tweets.
    """
    df = df.copy()

    # If 'retweet' isn't present, guess using "text starts with RT"
    if "retweet" not in df.columns:
        if "text" in df.columns:
            df["retweet"] = df["text"].astype(str).str.match(r"^RT")
        else:
            df["retweet"] = False

    # Coerce to clean boolean
    df["retweet"] = _coerce_retweet(df["retweet"])

    # Filter out rows that are retweets
    df = df[~df["retweet"]].copy()

    # Keep a clean column (all False now, since retweets are removed)
    df["retweet"] = False
    return df


# --------------------------------------------------------------
# PREPROCESS FUNCTION (Main entry point)
# --------------------------------------------------------------

@st.cache_data
def preprocess(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Clean tweets / posts dataset and return exploded bag-of-words.

    Output: one row per (tweet_id, term) with frequency.
            Columns:
            ['tweet_id', 'text', 'created_at', 'term', 'frequency']

    Steps:
    1. Remove retweets safely.
    2. Drop duplicates.
    3. Detect language, filter English only.
    4. Lowercase + stemming.
    5. Tokenize to term frequencies.
    6. Expand (tweet_id x term).
    """
    df = df_raw.copy()

    # 1. Remove retweets
    df = _filter_out_retweets(df)

    # 2. Drop duplicates by text & tweet_id
    for col in ["text", "tweet_id"]:
        if col not in df.columns:
            df[col] = ""
    df = df.drop_duplicates(subset=["text", "tweet_id"], keep="first")

    # 3. Ensure text is string
    df["text"] = df["text"].fillna("").astype(str)

    # 4. Language detection
    df["language"] = df["text"].apply(detect_language)

    # 5. Filter to English only
    df = df[df["language"] == "en"]

    # 6. Lowercase base text
    df["text"] = df["text"].str.lower()

    # 7. Stemming
    df["text_stemmed"] = df["text"].apply(stem_sentence)

    # 8. Tokenize to term frequency
    df["term_freq"] = df["text"].apply(tokenize_and_count)

    # 9. Expand to one row per token
    expanded_rows = []
    for _, row in df.iterrows():
        for term, freq in row["term_freq"].items():
            expanded_rows.append([
                row.get("tweet_id", None),
                row.get("text", ""),
                row.get("created_at", None),
                term,
                freq,
            ])

    result_df = pd.DataFrame(
        expanded_rows,
        columns=["tweet_id", "text", "created_at", "term", "frequency"],
    )

    return result_df


# --------------------------------------------------------------
# JOIN AND PER-TWEET SCORING
# --------------------------------------------------------------

@st.cache_data
def join_and_multiply_data(
    df_join: pd.DataFrame,
    dictionary: pd.DataFrame,
    old_df: pd.DataFrame,
    timestamp_format_macro: str = "ISO8601",
    extra_dict: pd.DataFrame | None = None,
    extra_driver: str | None = None,
) -> pd.DataFrame:
    """
    Take exploded term-frequency data (df_join),
    join with sentiment dictionary,
    aggregate to tweet-level scores,
    compute per-driver nets and Brand_Reputation,
    and return ONE ROW PER TWEET with all scores.

    Parameters
    ----------
    df_join : DataFrame
        Output of preprocess(): bag-of-words exploded.
        ['tweet_id','text','created_at','term','frequency']
    dictionary : DataFrame
        Has 'term' and columns like 'price_pos','price_neg',...
    old_df : DataFrame
        The original, pre-tokenized tweets. Must include 'tweet_id',
        'text', 'created_at' at minimum.
    timestamp_format_macro : str
        Passed to change_time_columns to enrich/normalize created_at.
    extra_dict : DataFrame or None
        Optional extra dictionary to merge on 'term'.
    extra_driver : str or None
        Optional new driver name (e.g. "safety") that behaves like other drivers.

    Returns
    -------
    tweet_scored_df : DataFrame
        Columns include:
        - tweet_id, text, created_at (+ year/month/etc. from change_time_columns)
        - *_pos, *_neg, *_net per subdriver
        - Value_Driver, Brand_Driver, Relationship_Driver, [extra_driver]_Driver
        - Brand_Reputation (unnormalized per tweet)
    """

    # 0. Merge extra_dict into dictionary if provided
    if extra_dict is not None:
        dictionary = pd.merge(dictionary, extra_dict, on="term", how="inner")

    # 1. Join exploded tweet-term-frequency with dictionary
    joined_df = pd.merge(df_join, dictionary, on="term", how="inner")

    # 2. Multiply frequency by each scoring column in the dictionary
    drives_list = dictionary.columns.tolist()
    if "term" in drives_list:
        drives_list.remove("term")

    for col in drives_list:
        joined_df[col] = joined_df["frequency"] * joined_df[col]

    # 3. Sum those scores at tweet level
    tweet_level_df = joined_df[["tweet_id"] + drives_list].groupby("tweet_id").sum().reset_index()

    # 4. Reattach original tweet metadata, plus derived time columns
    old_df_enriched = change_time_columns(old_df, format_arg=timestamp_format_macro)
    merged_df = pd.merge(old_df_enriched, tweet_level_df, on="tweet_id", how="inner")

    # --------------------------------------------------------
    # 5. Compute *_net for each subdriver and per-driver buckets
    # --------------------------------------------------------
    driver_groups = {
        "Value": ["price", "squality", "goodsq"],
        "Brand": ["cool", "exciting", "innov", "socresp"],
        "Relationship": ["comm", "friendly", "personalrel", "trust"],
    }
    if extra_driver is not None:
        driver_groups[extra_driver] = [extra_driver]

    # *_net = *_pos - *_neg for each subdriver
    for bucket, subs in driver_groups.items():
        for sub in subs:
            pos_col = f"{sub}_pos"
            neg_col = f"{sub}_neg"
            net_col = f"{sub}_net"

            if pos_col in merged_df.columns and neg_col in merged_df.columns:
                merged_df[net_col] = merged_df[pos_col] - merged_df[neg_col]
            else:
                # if missing, fill with 0 so downstream math doesn't break
                merged_df[net_col] = 0

    # Compute bucket-level driver score as mean of *_net columns
    for bucket, subs in driver_groups.items():
        net_cols = [f"{s}_net" for s in subs]
        bucket_col = f"{bucket}_Driver"
        merged_df[bucket_col] = merged_df[net_cols].mean(axis=1)

    # --------------------------------------------------------
    # 6. Compute Brand_Reputation at tweet level
    # --------------------------------------------------------
    components = ["Value_Driver", "Brand_Driver", "Relationship_Driver"]
    if extra_driver is not None:
        components.append(f"{extra_driver}_Driver")

    merged_df["Brand_Reputation"] = merged_df[components].sum(axis=1)

    # 7. Return row-per-tweet table
    return merged_df


# --------------------------------------------------------------
# TIME-AGGREGATED (for dashboard charts)
# --------------------------------------------------------------

def prepare_aggregate_dict(list_of_drives):
    """
    Build aggregation dictionary for compute_drives().
    tweet_id will become total_tweets.
    *_pos/*_neg will be summed.
    """
    aggregate_dict = {"tweet_id": "count"}
    for drive in list_of_drives:
        aggregate_dict[drive + "_pos"] = "sum"
        aggregate_dict[drive + "_neg"] = "sum"
    return aggregate_dict


@st.cache_data
def compute_drives(
    final_df: pd.DataFrame,
    granularity,
    extra_driver: str | None = None,
) -> pd.DataFrame:
    """
    Aggregate scores by a given granularity (e.g. 'month' or ['year','month']),
    compute *_net and driver buckets, normalize them, and return a time series.

    final_df should be the tweet_scored_df from join_and_multiply_data()
    (one row per tweet with *_pos/*_neg, *_net, *_Driver, etc.).
    """

    # which subdriver prefixes we expect
    dict_columns_ids = [
        "price", "squality", "goodsq",
        "cool", "exciting", "innov", "socresp",
        "comm", "friendly", "personalrel", "trust",
    ]
    if extra_driver is not None:
        dict_columns_ids.append(extra_driver)

    # build aggregation dict: sums for *_pos/_neg, "first" timestamp representative
    dict_to_aggregate = prepare_aggregate_dict(dict_columns_ids)
    dict_to_aggregate["created_at"] = "first"

    grouped_df = final_df.groupby(granularity).agg(dict_to_aggregate).reset_index()

    grouped_df = grouped_df.rename(columns={"tweet_id": "total_tweets"})

    driver_groups = {
        "Value": ["price", "squality", "goodsq"],
        "Brand": ["cool", "exciting", "innov", "socresp"],
        "Relationship": ["comm", "friendly", "personalrel", "trust"],
    }
    if extra_driver is not None:
        driver_groups[extra_driver] = [extra_driver]

    driver_columns = []

    # build *_net per subdriver and bucket drivers
    for bucket, subs in driver_groups.items():
        # *_net for each subdriver
        for sub in subs:
            pos_col = f"{sub}_pos"
            neg_col = f"{sub}_neg"
            net_col = f"{sub}_net"

            if pos_col in grouped_df.columns and neg_col in grouped_df.columns:
                grouped_df[net_col] = grouped_df[pos_col] - grouped_df[neg_col]
            else:
                grouped_df[net_col] = 0

            driver_columns.append(net_col)

        # bucket-level driver (e.g. Value_Driver)
        net_cols = [f"{s}_net" for s in subs]
        bucket_col = f"{bucket}_Driver"
        grouped_df[bucket_col] = grouped_df[net_cols].mean(axis=1)
        driver_columns.append(bucket_col)

    # Brand Reputation (unnormalized for the bucket)
    grouped_df["Brand Reputation"] = (
        grouped_df["Value_Driver"] +
        grouped_df["Brand_Driver"] +
        grouped_df["Relationship_Driver"] +
        (grouped_df[f"{extra_driver}_Driver"] if extra_driver else 0)
    )

    # Z-score normalize all score columns (per group of rows)
    all_score_cols = driver_columns + ["Brand Reputation"]
    for col in all_score_cols:
        mean = grouped_df[col].mean()
        std = grouped_df[col].std()
        grouped_df[col] = 0 if (std == 0 or pd.isna(std)) else (grouped_df[col] - mean) / std

    # chronological sort
    grouped_df.sort_values(by="created_at", inplace=True)

    # final column order
    final_cols = []
    if isinstance(granularity, str):
        final_cols.append(granularity)
    else:
        final_cols.extend(granularity)

    final_cols += ["created_at", "total_tweets"] + all_score_cols
    grouped_df = grouped_df[final_cols]

    return grouped_df


# --------------------------------------------------------------
# UTF-8 + LOOKER STUDIO EXPORT HELPERS
# --------------------------------------------------------------

def _format_created_at_for_looker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a copy of df where 'created_at' (if present) is:
    - coerced to datetime
    - timezone stripped
    - formatted as 'YYYY-MM-DD HH:MM:SS' (string)

    Looker Studio will auto-detect this as Date & Time.
    """
    if "created_at" not in df.columns:
        return df

    df = df.copy()

    # force to datetime
    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    # drop tz if present
    try:
        df["created_at"] = df["created_at"].dt.tz_localize(None)
    except Exception:
        pass

    # final string format
    df["created_at"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S")

    return df


def save_dataframe_utf8(df: pd.DataFrame, filename: str, filetype: str = "csv"):
    """
    Save df to disk with:
    - UTF-8 encoding (CSV/TXT)
    - Looker-friendly created_at formatting
    - Excel option for XLSX
    """
    filetype = filetype.lower()
    df_to_save = _format_created_at_for_looker(df)

    if filetype == "csv":
        df_to_save.to_csv(filename, index=False, encoding="utf-8")

    elif filetype == "txt":
        # write CSV-style text out to .txt in UTF-8
        with open(filename, "w", encoding="utf-8") as f:
            f.write(df_to_save.to_csv(index=False, encoding="utf-8"))

    elif filetype in ("xlsx", "xls"):
        df_to_save.to_excel(filename, index=False)

    else:
        raise ValueError(f"Unsupported filetype: {filetype}")

    print(f"✅ File saved as {filename} (UTF-8, Looker-ready dates)")


def get_utf8_download_buffer(df: pd.DataFrame) -> bytes:
    """
    Return UTF-8 encoded CSV bytes for Streamlit's st.download_button,
    with created_at already formatted for Looker.
    """
    df_to_send = _format_created_at_for_looker(df)

    buffer = io.StringIO()
    df_to_send.to_csv(buffer, index=False, encoding="utf-8")
    return buffer.getvalue().encode("utf-8")
