import streamlit as st
import pandas as pd
import backend as bk


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
df = pd.read_csv("data/default_clean.csv")
default_dictionary = pd.read_excel("data/default_dict.xlsx")


headLeft, headSpacer, headRight = st.columns([0.7,0.05,0.2])

headRight.markdown(" ")
headRight.markdown(" ")

headLeft.title("Brand Reputation App")
st.page_link("pages/app.py", label="Compute and Visualize Brand Reputation", icon="▶️")
st.markdown("---")
st.markdown(" ")
# Add integer input for example purposes
rename = False
user_input_integer = 1000
if headRight.checkbox("Edit"):
    rename = True
    # Create three columns
    col1, spacer, col2 = st.columns([1.2,0.1,0.8])
    user_input_integer = col1.number_input('Sample Size', min_value=100, value=1000, step=1)

    uploaded_file = col1.file_uploader("Choose a CSV file", type=("csv"))


    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        col1.write("Tweets Data Uploaded")

    # Add a horizontal line
    col1.markdown("---")

    col11, spacer11, col12 = col1.columns([1,0.1,1])

    

    # Add a download button for the Excel file

    svg_string = """
    <center>
    <svg width="50px" height="50px" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" fill="#000000" version="1.1" id="Capa_1" width="800px" height="800px" viewBox="0 0 395.001 395" xml:space="preserve">
    <g>
        <g>
            <path d="M322.852,0H72.15C32.366,0,0,32.367,0,72.15v250.7C0,362.634,32.367,395,72.15,395h250.701    c39.784,0,72.15-32.366,72.15-72.15V72.15C395.002,32.367,362.635,0,322.852,0z M370.002,322.85    c0,25.999-21.151,47.15-47.15,47.15H72.15C46.151,370,25,348.849,25,322.85V72.15C25,46.151,46.151,25,72.15,25h250.701    c25.999,0,47.15,21.151,47.15,47.15L370.002,322.85L370.002,322.85z"/>
            <path d="M197.501,79.908c-33.775,0-61.253,27.479-61.253,61.254c0,6.903,5.596,12.5,12.5,12.5c6.904,0,12.5-5.597,12.5-12.5    c0-19.99,16.263-36.254,36.253-36.254s36.253,16.264,36.253,36.254c0,11.497-8.314,19.183-22.01,30.474    c-12.536,10.334-26.743,22.048-26.743,40.67v40.104c0,6.902,5.597,12.5,12.5,12.5c6.903,0,12.5-5.598,12.5-12.5v-40.104    c0-6.832,8.179-13.574,17.646-21.381c13.859-11.426,31.106-25.646,31.106-49.763C258.754,107.386,231.275,79.908,197.501,79.908z"/>
            <path d="M197.501,283.024c-8.842,0-16.034,7.193-16.034,16.035c0,8.84,7.192,16.033,16.034,16.033    c8.841,0,16.034-7.193,16.034-16.033C213.535,290.217,206.342,283.024,197.501,283.024z"/>
        </g>
    </g>
    </svg>
    </center>
    """

    info_string = "You can upload an XLSX file with 3 columns and the same amount of rows of the current dictionary. &#013; Download the current dictionary to generate such data."

    col11.markdown("<span title='" + info_string + "'>" + svg_string + "</span>", unsafe_allow_html=True)

    col12.download_button(
        label="Default Dictionary",
        data=df.to_csv(index=False).encode(),
        file_name="default_dict.csv",
        mime="text/csv",
    )

    # Add a switch to the left column
    if col12.checkbox("Add New Driver"):
        uploaded_file2 = col1.file_uploader("Upload Extra Dictionary (Additional Driver)", type=("xlsx"))
        # Add a text input field below the second file upload
        text_input = col1.text_input("New Driver Name", "Custom")

        if uploaded_file2 is not None:
            df_add = pd.read_excel(uploaded_file2)
            col1.write("Additional Driver Dictionary Uploaded")
        else:
            df_add = pd.read_excel("data/default_additional_dictionary.xlsx")

    # Add a string input to the right column with a default value
    string_input = col2.text_input("Timestamp Format", "ISO8601")

    # Add three column selections to the right column with default values
    col_selection1 = col2.selectbox("Select ID", df.columns, index=1)
    col_selection2 = col2.selectbox("Select Timestamp", df.columns, index=2)
    col_selection3 = col2.selectbox("Select Text", df.columns, index=0)

    new_col_names = [col_selection1, col_selection2, col_selection3]

    no_error_renaming = True
    if len(set(new_col_names)) != 3:
        no_error_renaming = False
        col2.markdown('<span style="font-style: bold; color: red;">Error: Column names must be unique</span>', unsafe_allow_html=True)

if rename and no_error_renaming:
    st.markdown('<span style="font-style: bold; color: green;">Note: Edits to the tweets table are shown in the table below</span>', unsafe_allow_html=True)
    df.rename(columns={
        new_col_names[0]: "tweet_id-32876tjkdhsba",
        new_col_names[1]: "created_at-32876tjkdhsba",
        new_col_names[2]: "text-32876tjkdhsba"
    }, inplace=True)

    df.rename(columns={
    "tweet_id-32876tjkdhsba": "tweet_id",
    "created_at-32876tjkdhsba": "created_at",
    "text-32876tjkdhsba": "text"
    }, inplace=True)

st.table(df.head())

df.sample(user_input_integer).to_csv("data/cached_df.csv", index=False)
#df.to_csv("data/cached_df.csv", index=False)
default_dictionary.to_csv("data/cached_dictionary.csv", index=False)