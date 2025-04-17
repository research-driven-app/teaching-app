# Settings and Visualization '/pages/edit.py'

# Import relevant libraries
import streamlit as st
import pandas as pd
import backend as bk
from io import BytesIO

# Define the session states for the dictionaries
df = st.session_state['cached_df']
dictionary = st.session_state['cached_dictionary']
pattern = st.session_state['timestamp_pattern']

df_preprocessed = bk.preprocess(df)

df_joined = bk.join_and_multiply_data(df_preprocessed, 
                                      dictionary, 
                                      df, 
                                      timestamp_format_macro=pattern)

# Set the title of the page
st.title("Brand Reputation App")

st.page_link("app.py", label="Settings", icon="🔙")
st.markdown("---")
st.markdown(" ")


# Create three columns (first column with 40% width, 10% space and third column 50% width)
col1, spacer, col2 = st.columns([0.4,0.1,0.5])

# Create options for the column selector widget
options = ['Weekly', 'Monthly', 'Quarterly', 'Yearly']

# Create the selector widget in the first column
selection = col1.selectbox('Granularity', options, index=0)

# Weekly aggregation, monthly aggregation, quarterly aggregation, yearly aggregation. we save the selection in "gran" 
if selection == 'Weekly':
  gran = ['year', 'quarter', "month", "week"]

elif selection == 'Monthly':
  gran = ['year', 'quarter', "month"]

elif selection == 'Quarterly':
  gran = ['year', 'quarter']

else:
  gran = ['year']


# From the backend, apply the function "compute drives", taking in the text-mined dataframe and the aggregation
df_drivers = bk.compute_drives(df_joined, gran)

# Create a BytesIO buffer for the Excel file
output = BytesIO()
# Write the DataFrame to this buffer (to be downloaded later, if wanted)
df_drivers.to_excel(output, index=False)
# Seek to the beginning of the stream
output.seek(0)


# in the first column, set a download button to download drivers
col1.download_button(
    label="Download All Drivers",
    data=output.read(),
    file_name="computed_drivers.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)

# Define a list with all columns from our driver dataframe
columns = list(df_drivers.columns)

# Let user select which columns to plot
all_columns = df_drivers.columns.tolist()
selection2 = st.multiselect("Select drivers to plot:", all_columns, default='Brand Reputation')

# Filter the dataframe to show only selected columns
if selection2:
    st.line_chart(df_drivers[selection2])
else:
    st.warning("Please select at least one driver to display the plot.")

