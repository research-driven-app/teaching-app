import streamlit as st
import pandas as pd
import backend as bk
from io import BytesIO
import plotly.graph_objects as go

st.set_page_config(initial_sidebar_state="collapsed")

#df = pd.read_csv("data/cached_df.csv")

#dictionary = pd.read_csv("data/cached_dictionary.csv")

df = st.session_state['cached_df']
dictionary = st.session_state['cached_dictionary']
extra_dictionary = st.session_state['new_dictionary']
extra_drive = st.session_state['new_drive']
pattern = st.session_state['timestamp_pattern']

df_preprocessed = bk.preprocess(df)

df_joined = bk.join_and_multiply_data(df_preprocessed, 
                                      dictionary, 
                                      df, 
                                      timestamp_format_macro=pattern, 
                                      extra_dict=extra_dictionary)



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
  plotly_legend = dict(tickformat='%Y-%m-%d')
elif selection == 'Monthly':
  gran = ['year', 'quarter', "month"]
  plotly_legend = dict(tickformat='%Y-%m')
elif selection == 'Quarterly':
  gran = ['year', 'quarter']
  plotly_legend = dict(tickformat='%Y-%m')

else:
  gran = ['year']
  plotly_legend = dict(tickformat='%Y')



df_drives = bk.compute_drives(df_joined, gran, extra_driver=extra_drive)

# Create a BytesIO buffer for the Excel file
output = BytesIO()
# Write the DataFrame to this buffer
df_drives.to_excel(output, index=False)
# Seek to the beginning of the stream
output.seek(0)

col1.download_button(
    label="Download All Drives",
    data=output.read(),
    file_name="computed_drives.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
)


colums = list(df_drives.columns)
selection2 = col1.multiselect('Drives', colums , default=["Brand Reputation"])

# Create a Plotly graph object figure
fig = go.Figure()

# Add a line for each column in columns_to_plot
for column in selection2:
    fig.add_trace(go.Scatter(x=df_drives.index, y=df_drives[column], mode='lines', name=column))


# Update layout if needed
fig.update_layout(xaxis_title='Time',
                  yaxis_title='Drives',
                  xaxis=plotly_legend,
                  legend=dict(
                      x=0.5,  # Centers the legend above the plot
                      y=1.2,  # Places the legend above the plot area
                      xanchor='center',  # Centers the legend at the x position
                      yanchor='bottom',  # Anchors the legend's bottom at the y position
                      orientation='h'  # Optional: Makes the legend items align horizontally
                  )
)


col2.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
