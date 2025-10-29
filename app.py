import streamlit as st
import pandas as pd
from io import BytesIO
from typing import Tuple

########################################
# 0. Page config
########################################

st.set_page_config(page_title="Brand Reputation Teaching App")

########################################
# 1. Helpers for timestamp detection / normalization
########################################

COMMON_PATTERNS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%d/%m/%Y %H:%M:%S",
    "%d/%m/%Y %H:%M",
    "%d/%m/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%Y/%m/%d %H:%M:%S",
    "%Y/%m/%d %H:%M",
    "%Y/%m/%d",
]


def try_infer_datetime(series: pd.Series) -> Tuple[pd.Series, str | None, float]:
    """
    Try to parse a timestamp column into pandas datetime, assume incoming values are UTC,
    then we will convert to Europe/Paris.

    Returns (parsed_series, detected_format, success_rate).

    parsed_series: pandas Series (datetime64[ns, UTC] or NaT)
    detected_format: str or "inferred" or None
    success_rate: float in [0,1]
    """
    if series is None:
        return pd.Series(dtype="datetime64[ns]"), None, 0.0

    # Work on a sample of non-null values as strings
    sample = series.dropna().astype(str).head(20)

    # 1. Let pandas guess freely
    try:
        _ = pd.to_datetime(sample, errors='raise', infer_datetime_format=True, utc=True)
        full_parsed = pd.to_datetime(series, errors='coerce', infer_datetime_format=True, utc=True)
        ok_ratio = full_parsed.notna().mean()
        return full_parsed, "inferred", ok_ratio
    except Exception:
        pass

    # 2. Try known explicit formats
    for fmt in COMMON_PATTERNS:
        try:
            _ = pd.to_datetime(sample, format=fmt, errors='raise', utc=True)
            full_parsed = pd.to_datetime(series, format=fmt, errors='coerce', utc=True)
            ok_ratio = full_parsed.notna().mean()
            return full_parsed, fmt, ok_ratio
        except Exception:
            continue

    # 3. Fallback: best effort coercion
    fallback = pd.to_datetime(series, errors='coerce', utc=True)
    ok_ratio = fallback.notna().mean()
    return fallback, None, ok_ratio


def normalize_created_at(df: pd.DataFrame) -> Tuple[pd.DataFrame, str | None, float]:
    """
    Ensure df['created_at'] exists and is parsed to naive Europe/Paris time if possible.
    Returns (df_out, detected_format, ok_ratio).
    """
    if "created_at" not in df.columns:
        return df, None, 0.0

    parsed_dt_col, detected_fmt, ok_ratio = try_infer_datetime(df["created_at"])

    # If it has timezone info, convert to Europe/Paris, then drop tz to make it Excel-friendly
    if hasattr(parsed_dt_col.dt, "tz") and parsed_dt_col.dt.tz is not None:
        parsed_dt_col = (
            parsed_dt_col
            .dt.tz_convert("Europe/Paris")
            .dt.tz_localize(None)
        )
    else:
        # If it's naive already (no tz), just keep it as-is
        parsed_dt_col = parsed_dt_col.dt.tz_localize(None)

    df = df.copy()
    df["created_at"] = parsed_dt_col
    return df, detected_fmt, ok_ratio


def coerce_retweet_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Guarantee there's a boolean `retweet` column.
    If it's missing, create it (all False).
    If it's messy (e.g. "True"/"False", 1/0, yes/no),
    coerce it to clean True/False.
    """
    df = df.copy()

    if "retweet" not in df.columns:
        df["retweet"] = False
        return df

    raw = df["retweet"]

    if raw.dtype == bool:
        retweet_bool = raw

    elif pd.api.types.is_bool_dtype(raw):
        retweet_bool = raw.astype(bool)

    elif pd.api.types.is_integer_dtype(raw) or pd.api.types.is_float_dtype(raw):
        # Treat nonzero as True
        retweet_bool = raw.fillna(0).astype(int).astype(bool)

    else:
        # Assume object-like column with strings like "True", "FALSE", "0", "1", etc.
        retweet_bool = (
            raw.astype(str)
               .str.strip()
               .str.lower()
               .map({
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False,
                    "yes": True,
                    "no": False
               })
        )

    # Default any unmapped / NaN to False
    retweet_bool = retweet_bool.fillna(False).astype(bool)

    df["retweet"] = retweet_bool
    return df


def validate_required_schema(df: pd.DataFrame) -> list[str]:
    """
    Returns a list of missing required columns.
    Edit page and backend assume at least these 3.
    """
    required_cols = ["tweet_id", "created_at", "text"]
    return [c for c in required_cols if c not in df.columns]


########################################
# 2. Load initial data (fallback data)
########################################

# If these files don't exist, Streamlit will crash on first run.
# You can wrap these in try/except if you want graceful failure for students.
default_dictionary = pd.read_excel("data/default_dict.xlsx")
df = pd.read_csv("data/default_clean.csv")

# default UI state
user_input_integer = 1000
edit_mode = False

########################################
# 3. Header / nav
########################################

headLeft, headSpacer, headRight = st.columns([0.7, 0.05, 0.2])

headRight.markdown(" ")
headRight.markdown(" ")

headLeft.title("Brand Reputation App")

# Link to the next page
st.page_link(
    "pages/edit.py",
    label="Compute and Visualize Brand Reputation",
    icon="▶️"
)

st.markdown("---")
st.markdown(" ")


########################################
# 4. Editable settings panel
########################################

if headRight.checkbox("Edit"):
    edit_mode = True

    # Two-column layout while editing
    col1, spacer, col2 = st.columns([1.2, 0.1, 0.8])

    # =========================
    # 4a. Upload CSV + sample size
    # =========================
    uploaded_file = col1.file_uploader("Choose a CSV file", type=("csv",))

    user_input_integer = col1.number_input(
        "Sample Size",
        min_value=100,
        value=1000,
        step=1
    )

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        col1.write("Tweets Data Uploaded")

    col1.markdown("---")

    # =========================
    # 4b. Dictionary download helper block
    # =========================
    col11, spacer11, col12 = col1.columns([1, 0.1, 1])

    # Tooltip icon (SVG with hover text)
    with open("svg_question_mark.txt", "r") as file:
        svg_string = file.read()

    info_string = (
        "You can upload an XLSX file with 3 columns and the same amount of "
        "rows of the current dictionary. &#013; "
        "Download the current dictionary to generate such data."
    )
    svg_string = svg_string.replace("INSERT_HERE", info_string)
    col11.markdown(svg_string, unsafe_allow_html=True)

    # Prepare default dictionary as downloadable Excel
    dict_buffer = BytesIO()
    default_dictionary.to_excel(dict_buffer, index=False)
    dict_buffer.seek(0)

    col12.download_button(
        label="Default Dictionary",
        data=dict_buffer.read(),
        file_name="default_dict.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # =========================
    # 4c. Column mapping widgets
    # =========================
    # let the user map: tweet_id / created_at / text
    # We try to guess: ID -> 1st col? Timestamp -> 2nd? Text -> 0th?
    col_selection_id = col2.selectbox(
        "Select ID",
        df.columns,
        index=1 if len(df.columns) > 1 else 0
    )
    col_selection_time = col2.selectbox(
        "Select Timestamp",
        df.columns,
        index=2 if len(df.columns) > 2 else 0
    )
    col_selection_text = col2.selectbox(
        "Select Text",
        df.columns,
        index=0
    )

    chosen_cols = [col_selection_id, col_selection_time, col_selection_text]

    unique_mapping = len(set(chosen_cols)) == 3
    if not unique_mapping:
        col2.markdown(
            '<span style="color: red;"><b>Error</b>: Column names must be unique</span>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # =========================
    # 4d. Apply renaming + datetime parsing preview
    # =========================
    if unique_mapping:
        st.markdown(
            '<span style="color: green;"><b>Note</b>: Edits to the tweets table are shown in the table below</span>',
            unsafe_allow_html=True
        )

        # (1) Rename selected columns to internal standard names.
        # We do a two-hop rename so we don't collide if user already has "tweet_id".
        temp_rename = {
            col_selection_id: "tweet_id___tmp_internal",
            col_selection_time: "created_at___tmp_internal",
            col_selection_text: "text___tmp_internal",
        }
        df = df.rename(columns=temp_rename)

        final_rename = {
            "tweet_id___tmp_internal": "tweet_id",
            "created_at___tmp_internal": "created_at",
            "text___tmp_internal": "text",
        }
        df = df.rename(columns=final_rename)

        # (2) Normalize timestamp to naive Europe/Paris
        df, detected_fmt, ok_ratio = normalize_created_at(df)

        # (3) Ensure boolean `retweet` column always exists/clean
        df = coerce_retweet_column(df)

        # (4) Give user feedback about datetime parsing
        if detected_fmt is None:
            st.warning(
                f"⚠️ I couldn't confidently detect a single timestamp format. "
                f"Parsed about {ok_ratio*100:.1f}% of rows. "
                f"Unparsed rows will show up as blank dates."
            )
        else:
            st.success(
                f"✅ Detected timestamp format: {detected_fmt}. "
                f"Parsed {ok_ratio*100:.1f}% of rows."
            )

        # (5) Preview standardized columns
        preview_cols = [c for c in ["tweet_id", "created_at", "text", "retweet"] if c in df.columns]
        st.write(df[preview_cols].head())


########################################
# 5. Sample limiting + final df_to_cache
########################################

# Even if edit mode is off, we still want sane defaults:
# - normalize created_at to datetime (Paris)
# - ensure retweet exists/clean
df, _, _ = normalize_created_at(df)
df = coerce_retweet_column(df)

# Respect the chosen / default sample size
user_input_integer = min(len(df), int(user_input_integer))
df_to_cache = df.head(user_input_integer).copy()

# Basic schema validation before we show & cache
missing_required = validate_required_schema(df_to_cache)

if missing_required:
    st.error(
        "Your data is missing required columns: "
        + ", ".join(missing_required)
        + ". Please use Edit mode to map columns correctly."
    )
else:
    # Show preview table
    st.table(df_to_cache.head())

    # Save for next page
    st.session_state["cached_df"] = df_to_cache
    st.session_state["cached_dictionary"] = default_dictionary

    # We no longer need to store a timestamp pattern, since we parsed `created_at`.
    st.session_state["timestamp_pattern"] = None
