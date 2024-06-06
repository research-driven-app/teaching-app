import streamlit as st
import pandas as pd


def changeGran(granularity, drives):
  # add code to change drives df to desired granularity
  drives_granularity = drives
  return drives_granularity

st.set_page_config(initial_sidebar_state="collapsed")
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
  df = pd.read_excel("data/weekly_gran.xlsx")
elif selection == 'Monthly':
  df = pd.read_excel("data/monthly_gran.xlsx")
elif selection == 'Quarterly':
  df = pd.read_excel("data/quarterly_gran.xlsx")
else:
  df = pd.read_excel("data/yearly_gran.xlsx")

col1.download_button(
    label="Download All Drives",
    data=df.to_csv(index=False).encode(),
    file_name="default_dict.csv",
    mime="text/csv",
)

colums = list(df.columns)
selection2 = col1.multiselect('Drives', colums , default=["Brand Reputation"])


df = df[selection2]
df = changeGran(selection, df)
col2.line_chart(df)
