# Settings and Visualization '/pages/edit.py'

import streamlit as st
import pandas as pd
from io import BytesIO
import backend as bk


# --- helper to make datetime columns Excel-safe ---
def make_tz_naive(series: pd.Series) -> pd.Series:
    """
    Ensure a datetime-like Series is timezone-naive and safe for Excel export.
    """
    if not pd.api.types.is_datetime64_any_dtype(series):
        return series

    s = pd.to_datetime(series, errors="coerce")

    if s.dt.tz is not None:
        s = (
            s
            .dt.tz_convert("Europe/Paris")
            .dt.tz_localize(None)
        )
    else:
        try:
            s = s.dt.tz_localize(None)
        except Exception:
            pass

    return s


# ========== DATA PREP SECTION ==========

# Read from session_state (set in app.py)
df = st.session_state["cached_df"]
dictionary = st.session_state["cached_dictionary"]
pattern = st.session_state["timestamp_pattern"]

# Preprocess tweets (cleaning, language, stemming, token freq)
df_preprocessed = bk.preprocess(df)

# Join tweets with dictionary weights and timestamps
df_joined = bk.join_and_multiply_data(
    df_join=df_preprocessed,
    dictionary=dictionary,
    old_df=df,
    timestamp_format_macro=pattern,
    extra_dict=None,
    extra_driver=None,
)

# ========== PAGE LAYOUT / UI ==========

st.title("Brand Reputation App")

st.page_link("app.py", label="Settings", icon="🔙")
st.markdown("---")
st.markdown(" ")

# Two-column layout with spacer
col1, spacer, col2 = st.columns([0.4, 0.1, 0.5])

# Granularity options
options = ["Weekly", "Monthly", "Quarterly", "Yearly"]
selection = col1.selectbox("Granularity", options, index=0)

# Map dropdown selection to list of time columns we group by
if selection == "Weekly":
    gran = ["year", "quarter", "month", "week"]
elif selection == "Monthly":
    gran = ["year", "quarter", "month"]
elif selection == "Quarterly":
    gran = ["year", "quarter"]
else:  # "Yearly"
    gran = ["year"]

# Compute the reputation drivers time series / table
df_drivers = bk.compute_drives(df_joined, gran, extra_driver=None)

# ========== DOWNLOAD: PER-TWEET SCORES (Excel) ==========

st.markdown("### ⬇️ Download all tweets with individual scores")

st.write(
    "This Excel file contains one row per tweet/post with its driver scores "
    "(price_pos, price_neg, *_net, Value_Driver, Brand_Driver, Relationship_Driver, "
    "and Brand_Reputation)."
)

# Prepare Excel-safe copy
df_tweets_export = df_joined.copy()

# Sanitize any datetime columns
for c in df_tweets_export.columns:
    df_tweets_export[c] = make_tz_naive(df_tweets_export[c])

# Write Excel to memory
tweets_output = BytesIO()
df_tweets_export.to_excel(tweets_output, index=False)
tweets_output.seek(0)

# Download button
st.download_button(
    label="Download all tweets with scores (Excel)",
    data=tweets_output.read(),
    file_name="tweets_with_scores.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ========== DOWNLOAD: AGGREGATED DRIVERS (Excel) ==========

st.markdown("### ⬇️ Download aggregated time-series data")

# Make copy to avoid mutating the dataframe used for plotting
df_to_export = df_drivers.copy()

# Sanitize datetime columns for Excel
for c in df_to_export.columns:
    df_to_export[c] = make_tz_naive(df_to_export[c])

# Write Excel to memory
agg_output = BytesIO()
df_to_export.to_excel(agg_output, index=False)
agg_output.seek(0)

# Download button
col1.download_button(
    label="Download time-series drivers (Excel)",
    data=agg_output.read(),
    file_name="computed_drivers.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ========== PLOTTING ==========

# Let the user choose which columns to visualize
all_columns = df_drivers.columns.tolist()

selection2 = st.multiselect(
    "Select drivers to plot:",
    all_columns,
    default="Brand Reputation",
)

if selection2:
    # Sort by created_at if available so the chart uses chronological order
    if "created_at" in df_drivers.columns:
        df_plot = df_drivers.sort_values("created_at")
    else:
        df_plot = df_drivers

    st.line_chart(df_plot[selection2])
else:
    st.warning("Please select at least one driver to display the plot.")
