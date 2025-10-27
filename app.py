# FRONTEND PAGE 'app.py'

import streamlit as st
import pandas as pd
from io import BytesIO
import backend as bk  # you were importing this before; keeping it for consistency

########################################
# 1. Helpers for timestamp detection
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

def try_infer_datetime(series: pd.Series):
    """
    Try to parse a timestamp column into pandas datetime (timezone-aware in UTC).
    Returns (parsed_series, detected_format, success_rate).

    parsed_series: pandas Series (datetime64[ns] (UTC) or NaT)
    detected_format: str or None
    success_rate: float in [0,1]
    """
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


########################################
# 2. Default state / initial data
########################################

# Load default dictionary and default tweets data
default_dictionary = pd.read_excel("data/default_dict.xlsx")
df = pd.read_csv("data/default_clean.csv")

rename = False
user_input_integer = 1000
current_new_drive = None  # keeping this placeholder from your original code

# Page Title

st.set_page_config(page_title="Brand Reputation Teaching App")

# Page header layout
headLeft, headSpacer, headRight = st.columns([0.7, 0.05, 0.2])

headRight.markdown(" ")
headRight.markdown(" ")

headLeft.title("Brand Reputation App")

# Link to the next page
st.page_link("pages/edit.py", label="Compute and Visualize Brand Reputation", icon="▶️")
st.markdown("---")
st.markdown(" ")

########################################
# 3. Editable settings panel
########################################

if headRight.checkbox("Edit"):
    rename = True

    # Split layout for upload + settings
    col1, spacer, col2 = st.columns([1.2, 0.1, 0.8])

    # Upload CSV
    uploaded_file = col1.file_uploader("Choose a CSV file", type=("csv"))

    user_input_integer = col1.number_input(
        'Sample Size',
        min_value=100,
        value=1000,
        step=1
    )

    # If user uploaded their own data, use it
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        col1.write("Tweets Data Uploaded")

    # Horizontal rule
    col1.markdown("---")

    # Two subcolumns for dictionary download + helper text
    col11, spacer11, col12 = col1.columns([1, 0.1, 1])

    # Info icon (your existing SVG logic)
    with open('svg_question_mark.txt', 'r') as file:
        svg_string = file.read()

    info_string = (
        "You can upload an XLSX file with 3 columns and the same amount of "
        "rows of the current dictionary. &#013; "
        "Download the current dictionary to generate such data."
    )
    svg_string = svg_string.replace("INSERT_HERE", info_string)

    col11.markdown(svg_string, unsafe_allow_html=True)

    # Prepare default dictionary as downloadable Excel
    output = BytesIO()
    default_dictionary.to_excel(output, index=False)
    output.seek(0)

    col12.download_button(
        label="Default Dictionary",
        data=output.read(),
        file_name="default_dict.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Column mapping UI:
    # Let the user tell us which column is ID / Timestamp / Text
    col_selection1 = col2.selectbox("Select ID", df.columns, index=1 if len(df.columns) > 1 else 0)
    col_selection2 = col2.selectbox("Select Timestamp", df.columns, index=2 if len(df.columns) > 2 else 0)
    col_selection3 = col2.selectbox("Select Text", df.columns, index=0)

    new_col_names = [col_selection1, col_selection2, col_selection3]

    no_error_renaming = True
    if len(set(new_col_names)) != 3:
        no_error_renaming = False
        col2.markdown(
            '<span style="color: red;"><b>Error</b>: Column names must be unique</span>',
            unsafe_allow_html=True
        )

    st.markdown("---")

    ########################################
    # 4. Apply renaming, detect datetime, preview
    ########################################

    if rename and no_error_renaming:
        st.markdown(
            '<span style="color: green;"><b>Note</b>: Edits to the tweets table are shown in the table below</span>',
            unsafe_allow_html=True
        )

        # Temporary unique names (avoids collisions during rename)
        df.rename(columns={
            new_col_names[0]: "tweet_id-32876tjkdhsba",
            new_col_names[1]: "created_at-32876tjkdhsba",
            new_col_names[2]: "text-32876tjkdhsba"
        }, inplace=True)

        # Final standard names
        df.rename(columns={
            "tweet_id-32876tjkdhsba": "tweet_id",
            "created_at-32876tjkdhsba": "created_at",
            "text-32876tjkdhsba": "text"
        }, inplace=True)

        # --- NEW: detect and normalize datetime
        parsed_dt_col, detected_fmt, ok_ratio = try_infer_datetime(df['created_at'])

        # Convert UTC → Europe/Paris and drop tz to keep downstream happy / Excel-safe
        if hasattr(parsed_dt_col.dt, "tz") and parsed_dt_col.dt.tz is not None:
            parsed_dt_col = (
                parsed_dt_col
                .dt.tz_convert('Europe/Paris')
                .dt.tz_localize(None)
            )

        df['created_at'] = parsed_dt_col

        # Give feedback to the user
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

        # Show a small preview with normalized datetimes
        st.write(df[['tweet_id', 'created_at', 'text']].head())

########################################
# 5. Limit sample size and show table
########################################

user_input_integer = min(len(df), user_input_integer)
df_to_cache = df.head(user_input_integer)

# Preview table in the UI
st.table(df_to_cache.head())

########################################
# 6. Save state for the next page
########################################

# The next page (edit.py) will read these
st.session_state['cached_df'] = df_to_cache
st.session_state['cached_dictionary'] = default_dictionary

# We no longer need to pass around a free-text timestamp pattern. The data is already parsed.
st.session_state['timestamp_pattern'] = None
