# Settings and Visualization '/pages/edit.py'

import streamlit as st
import pandas as pd
import backend as bk
from io import BytesIO

# --- helper to make datetime columns Excel-safe ---
def make_tz_naive(series: pd.Series) -> pd.Series:
    """
    Ensure a datetime-like Series is timezone-naive and safe for Excel export.

    Rules:
    - If the Series is not datetime at all, return as-is.
    - If it's tz-aware: convert to Europe/Paris, then drop tz.
    - If it's tz-naive: just explicitly drop tz info (tz_localize(None)).
    """
    # If it's not datetime-ish, leave it untouched
    if not pd.api.types.is_datetime64_any_dtype(series):
        return series

    # Make sure it's datetime dtype (handles strings just in case)
    s = pd.to_datetime(series, errors='coerce')

    # If it's timezone-aware, .dt.tz is not None
    if s.dt.tz is not None:
        s = (
            s
            .dt.tz_convert('Europe/Paris')  # convert to local time
            .dt.tz_localize(None)           # then drop tz info
        )
    else:
        # It's tz-naive, but Excel still doesn't like tz-aware types,
        # so just enforce tz_localize(None) to be explicit
        s = s.dt.tz_localize(None)

    return s


# ========== DATA PREP SECTION ==========

# Read from Streamlit session_state (these must have been set in a previous page)
df = st.session_state['cached_df']
dictionary = st.session_state['cached_dictionary']
pattern = st.session_state['timestamp_pattern']

# Preprocess tweets (cleaning, language, stemming, token freq)
df_preprocessed = bk.preprocess(df)

# Join tweets with dictionary weights and timestamps
df_joined = bk.join_and_multiply_data(
    df_preprocessed,
    dictionary,
    df,
    timestamp_format_macro=pattern
)

# ========== PAGE LAYOUT / UI ==========

st.title("Brand Reputation App")

st.page_link("app.py", label="Settings", icon="🔙")
st.markdown("---")
st.markdown(" ")

# Two-column layout with spacer
col1, spacer, col2 = st.columns([0.4, 0.1, 0.5])

# Granularity options
options = ['Weekly', 'Monthly', 'Quarterly', 'Yearly']
selection = col1.selectbox('Granularity', options, index=0)

# Map dropdown selection to the list of time columns we group by
if selection == 'Weekly':
    gran = ['year', 'quarter', 'month', 'week']
elif selection == 'Monthly':
    gran = ['year', 'quarter', 'month']
elif selection == 'Quarterly':
    gran = ['year', 'quarter']
else:  # 'Yearly'
    gran = ['year']

# Compute the reputation drivers time series / table
df_drivers = bk.compute_drives(df_joined, gran)

# ========== EXPORT TO EXCEL (with timezone-safe timestamps) ==========

# We'll make a copy to avoid mutating df_drivers, because df_drivers is also used for plotting
df_to_export = df_drivers.copy()

# Walk through every column and sanitize datetime columns so Excel is happy
for col in df_to_export.columns:
    df_to_export[col] = make_tz_naive(df_to_export[col])

# Write cleaned data to an in-memory Excel file
output = BytesIO()
df_to_export.to_excel(output, index=False)
output.seek(0)

# Download button
col1.download_button(
    label="Download All Drivers",
    data=output.read(),
    file_name="computed_drivers.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# ========== PLOTTING ==========

# Let the user choose which columns to visualize
all_columns = df_drivers.columns.tolist()

selection2 = st.multiselect(
    "Select drivers to plot:",
    all_columns,
    default='Brand Reputation'
)

if selection2:
    # Streamlit will draw a line chart for each selected column over row order.
    # Note: If you want x-axis to be time, include a time column in df_drivers and sort before plotting.
    st.line_chart(df_drivers[selection2])
else:
    st.warning("Please select at least one driver to display the plot.")
