# It is helpful to familiarize with the basic functionalities of streamlit


# Before starting, we import the relevant packages

# We import streamlit 
import streamlit as st

# Since there are data operations that need to be done, we also import pandas
import pandas as pd

# Since there is a backend part, we also import the backend
import backend as bk


# This package will help us with the data buffering later
from io import BytesIO

# Default options are given here. In our case, we define a default text-mining dictionary, that we keep in the folder /data/
default_dictionary = pd.read_excel("data/default_dict.xlsx")

# This next line helps us with layouting, we define three columns. The one on the left takes 70%, then there is a small gap of 5%, and then the rest takes 20%
headLeft, headSpacer, headRight = st.columns([0.7,0.05,0.2])


# We want to push the content of the columns down, therefore we take two whitespaces
headRight.markdown(" ")
headRight.markdown(" ")


# This is the title of our app, "Brand Reputation App", it is set on the left on top "headLeft.title("name of the app goes here")"
headLeft.title("Brand Reputation App")

# Here, we link to the next page, this will execute the script, where the backend script (text-mining) would be executed
st.page_link("pages/app.py", label="Compute and Visualize Brand Reputation", icon="▶️")
st.markdown("---")
st.markdown(" ")

#Here we define the user input variables and their default values
rename = False
user_input_integer = 1000
current_pattern = "ISO8601"
current_new_drive = None
df_add = None
df = pd.read_csv("data/default_clean.csv")

### Here the editing of the settings on the first page is triggered (checkbox: Edit)
if headRight.checkbox("Edit"):
    rename = True
    # Create three columns (larger column for file upload, smaller column for settings)
    col1, spacer, col2 = st.columns([1.2,0.1,0.8])
    # Create a user input widget in the first column, setting the sample size, along with the minimum and default value
    user_input_integer = col1.number_input('Sample Size', min_value=100, value=1000, step=1)

    uploaded_file = col1.file_uploader("Choose a CSV file", type=("csv"))


    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        col1.write("Tweets Data Uploaded")

    # Add a horizontal line
    col1.markdown("---")

    col11, spacer11, col12 = col1.columns([1,0.1,1])

    

    # Add a download button for an Excel file
    with open('svg_question_mark.txt', 'r') as file:
        svg_string = file.read()

    info_string = "You can upload an XLSX file with 3 columns and the same amount of rows of the current dictionary. &#013; Download the current dictionary to generate such data."
    svg_string = svg_string.replace("INSERT_HERE", info_string)

    col11.markdown(svg_string, unsafe_allow_html=True)

    # Create a BytesIO buffer for the Excel file
    output = BytesIO()
    # Write the dictionary to this Excel buffer
    default_dictionary.to_excel(output, index=False)
    # Seek to the beginning of the stream

    output.seek(0)
    
    # add a download button for the default dictionary (defined above)

    col12.download_button(
        label="Default Dictionary",
        data=output.read(),
        file_name="default_dict.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    # Add a checkbox to the left column if new driver should be added
    if col12.checkbox("Add New Driver"):
        uploaded_file2 = col1.file_uploader("Upload Extra Dictionary (Additional Driver)", type=("xlsx"))
        # Add a text input field below the second file upload
        text_input = col1.text_input("New Driver Name", "Custom")

        if uploaded_file2 is not None:
            df_add = pd.read_excel(uploaded_file2)
            if "term" not in df_add.columns:
                col1.markdown('<span style="color: red;">Error: At least one column name must be "term"</span>', unsafe_allow_html=True)
            elif len(df_add.columns) != 3:
                col1.markdown('<span style="color: red;"><b>Error</b>: You should upload a dictionary with exactly 3 columns: term, ANY-STRING_pos, ANY-STRING_neg</span>', unsafe_allow_html=True)
            elif bk.check_columns_for_neg_suffix(df_add, "_neg"):
                col1.markdown('<span style="color: red;"><b>Error</b>: You should upload a dictionary with one of the columns called ANY-STRING_neg</span>', unsafe_allow_html=True)
            elif bk.check_columns_for_neg_suffix(df_add, "_pos"):
                col1.markdown('<span style="color: red;"><b>Error</b>: You should upload a dictionary with one of the columns called ANY-STRING_pos</span>', unsafe_allow_html=True)
            elif len(df_add) != len(default_dictionary):
                col1.markdown('<span style="color: red;"><b>Error</b>: You should upload a dictionary with unique terms in same amount of the current dictionary.</span>', unsafe_allow_html=True)
            else:
                col1.write('<span style="color: green;"><b>Success</b>: Additional Driver Dictionary Uploaded and Validated</span>', unsafe_allow_html=True)
                current_new_drive = text_input
            
                for col in df_add.columns:
                    if "_neg" in col:
                        df_add.rename(columns={col: current_new_drive + "_neg"}, inplace=True)
                    elif "_pos" in col:
                        df_add.rename(columns={col: current_new_drive + "_pos"}, inplace=True)
            

    # Add a string input to the right column with a default value for timefram
    current_pattern = col2.text_input("Timestamp Format", "ISO8601")

    # Add three column selections to the right column with default values (text in the first column, ID in the second, timestamp in the third column)
    col_selection1 = col2.selectbox("Select ID", df.columns, index=1)
    col_selection2 = col2.selectbox("Select Timestamp", df.columns, index=2)
    col_selection3 = col2.selectbox("Select Text", df.columns, index=0)

    new_col_names = [col_selection1, col_selection2, col_selection3]

    no_error_renaming = True
    if len(set(new_col_names)) != 3:
        no_error_renaming = False
        col2.markdown('<span style="color: red;"><b>Error</b>: Column names must be unique</span>', unsafe_allow_html=True)

    st.markdown("---")

if rename and no_error_renaming:
    st.markdown('<span style="color: green;"><b>Note</b>: Edits to the tweets table are shown in the table below</span>', unsafe_allow_html=True)
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

user_input_integer = min(len(df), user_input_integer)
df_to_cache = df.head(user_input_integer)

#Shows a preview of the table
st.table(df_to_cache.head())


# by using session stats, we can save the values of the variables chosen, so that, if we re-run the script, the original values are kept
# for example, if we have chosen an additional driver to add, it stays saved in the app during that session

st.session_state['timestamp_pattern'] = current_pattern
st.session_state['new_drive'] = current_new_drive
st.session_state['new_dictionary'] = df_add

st.session_state['cached_df'] = df_to_cache
st.session_state['cached_dictionary'] = default_dictionary

#df.sample(user_input_integer).to_csv("data/cached_df.csv", index=False)
#df.to_csv("data/cached_df.csv", index=False)
#default_dictionary.to_csv("data/cached_dictionary.csv", index=False)
