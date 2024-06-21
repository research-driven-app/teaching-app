import streamlit as st
import pandas as pd
import backend as bk
st.set_page_config(initial_sidebar_state="collapsed")

df = pd.read_csv("data/cached_df.csv")

dictionary = pd.read_csv("data/cached_dictionary.csv")

df_preprocessed = bk.preprocess(df)

df_joined = bk.join_and_multiply_data(df_preprocessed, dictionary, df)



st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)


st.title("Brand Reputation App")

st.page_link("app.py", label="Settings", icon="🔙")
st.markdown("---")
st.markdown(" ")


# Create three columns
col1, spacer, col2 = st.columns([0.4,0.1,0.5])

options = ['Weekly', 'Monthly', 'Quarterly', 'Yearly']
selection = col1.selectbox('Granularity', options, index=0)

if selection == 'Weekly':
  gran = ['year', 'quarter', "month", "week"]
elif selection == 'Monthly':
  gran = ['year', 'quarter', "month"]
elif selection == 'Quarterly':
  gran = ['year', 'quarter']
else:
  gran = ['year']


df_drives = bk.compute_drives(df_joined, gran)

col1.download_button(
    label="Download All Drives",
    data=df_drives.to_csv(index=False).encode(),
    file_name="drives_download.csv",
    mime="text/csv",
)


colums = list(df_drives.columns)
selection2 = col1.multiselect('Drives', colums , default=["Brand Reputation"])


df_drives = df_drives[selection2]

col2.line_chart(df_drives)
