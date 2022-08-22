from importlib.metadata import requires
from re import sub
import streamlit as st
import pandas as pd
import requests
from streamlit_option_menu import option_menu
from streamlit_tags import st_tags, st_tags_sidebar

from st_aggrid import AgGrid
from st_aggrid.grid_options_builder import GridOptionsBuilder
from st_aggrid.shared import JsCode
from st_aggrid import GridUpdateMode, DataReturnMode

from dashboard_utils.gui import keyboard_to_url
from dashboard_utils.gui import load_keyboard_class

###

if 'widen' not in st.session_state:
    layout = 'centered'
else:
    layout = 'wide' if st.session_state.widen else 'centered'

###

st.set_page_config(layout=layout,page_title='Zero-Shot Text Classifier',page_icon='🤗')

###

load_keyboard_class()

###

if not 'valid_inputs_received' in st.session_state:
    st.session_state['valid_inputs_received'] = False

### Title for main and sidebar

c1, c2 = st.columns([0.4, 2])

with c1:
    st.image('https://raw.githubusercontent.com/CharlyWargnier/zero-shot-classifier/main/logo.png',width=110)
with c2:
    st.caption('')
    st.title('Zero-Shot Text Classifier')

st.sidebar.image('https://raw.githubusercontent.com/CharlyWargnier/zero-shot-classifier/main/30days_logo.png')
st.sidebar.write('')

st.write('')
st.markdown('''
Classify keyphrases fast and on-the-fly with this mighty app. No ML training needed!

Create classifying labels (e.g. `Positive`, `Negative` and `Neutral`), paste your keyphrases, and you're off!
''')
st.write('')

### Sidebar

with st.sidebar:
    selected = option_menu(
        '',
        ['Demo (5 phrases max)', 'Unlocked Mode'],
        icons=['bi-joystick','bi-key-fill'],
        menu_icon='',
        default_index=0
    )

### 

st.sidebar.header('Shortcuts')
st.sidebar.write('<span class="kbdx">G</span> &nbsp; GitHub', unsafe_allow_html=True)
st.sidebar.write('<span class="kbdx">&thinsp;.&thinsp;</span> &nbsp; GitHub Dev (VS Code)', unsafe_allow_html=True)

###

st.sidebar.markdown('---')
st.sidebar.header('About')
st.sidebar.markdown('''
App created by [Datachaz](https://twitter.com/DataChaz) using 🎈[Streamlit](https://streamlit.io/) and [HuggingFace](https://huggingface.co/inference-api)'s [Distilbart-mnli-12-3](https://huggingface.co/valhalla/distilbart-mnli-12-3) model.

[Streamlit](https://streamlit.io) is a Python library that allows the creation of interactive, data-driven web applications in Python.
''')
st.sidebar.header('Resources')
st.sidebar.markdown('''
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Cheat sheet](https://docs.streamlit.io/library/cheatsheet)
- [Book](https://www.amazon.com/dp/180056550X) (Getting Started with Streamlit for Data Science)
- [Blog](https://blog.streamlit.io/how-to-master-streamlit-for-data-science/) (How to master Streamlit for data science)
''')
st.sidebar.header('Deploy')
st.sidebar.markdown('You can quickly deploy Streamlit apps using [Streamlit Cloud](https://streamlit.io/cloud) in just a few clicks.')

### interaction

def main():
    st.caption('')

if selected == 'Demo (5 phrases max)':
    API_KEY = st.secrets['API_KEY']
    API_URL = ('https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3')
    headers = {'Authorization': f'Bearer {API_KEY}'}
    
    with st.form(key='my_form'):
        multiselectComponent = st_tags(
            label='',
            text='Add labels - 3 max',
            value=['Transactional', 'Informational', 'Navigational'],
            suggestions=[
                'Transactional', 'Informational', 'Navigational', 
                'Positive', 'Negative', 'Neutral'
            ],
            maxtags=3,
        )
        new_line = '\n'
        nums = {
            'I want to buy something in this store',
            'How to ask a question about a product',
            'Request a refund through the Google Play store',
            'I have a broken screen, what should I do?',
            'Can I have the link to the product?',
        }
        sample = f'{new_line.join(map(str, nums))}'
        linesDeduped2 = []

        MAX_LINES = 5
        text = st.text_area(
            'Enter keyphrases to classify',
            sample,
            height=200,
            key=2,
            help='At least two keyphrases for the classfier to woek, one per line, '
            + str(MAX_LINES)
            + 'keyphrases max as part of the demo'
        )
        lines = text.split('\n')
        linesList = []
        for x in lines:
            linesList.append(x)
        linesList = list(dict.fromkeys(linesList)) # Remove dupes
        linesList = list(filter(None, linesList)) # Remove empty

        if len(linesList) > MAX_LINES:
            st.info(
                f'❄️  Only the first '
                + str(MAX_LINES)
                + " keyprases will be reviewed. Unlock that limit by switching to 'Unlocked Mode'"
            )
        linesList = linesList[:MAX_LINES]

        submit_button = st.form_submit_button(label='Submit')

    if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

    elif submit_button and not text:
        st.warning('❄️ There is no keyphrases to classify')
        st.session_state.value_inputs_received = False
        st.stop()

    elif submit_button and not multiselectComponent:
        st.warning('❄️ You have not added any labels, please add some! ')
        st.session_state.value_inputs_received = False
        st.stop()
    
    elif submit_button and len(multiselectComponent) == 1:
        st.warning('❄️ Please make sure to add at least two labels for classification')
        st.session_state.value_inputs_received = False
        st.stop()

    elif submit_button or st.session_state.value_inputs_received:
        if submit_button:
            st.session_state.value_inputs_received = True

        def query(payload):
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()
        
        listtest = ['I want a refund', 'I have a question']
        listToAppend = []
        for row in linesList:
            output2 = query(
                {
                    'inputs': row,
                    'parameters': {'candidate_labels': multiselectComponent},
                    'options': {'wait_for_model': True}
                }
            )
            listToAppend.append(output2)
            df = pd.DataFrame.from_dict(output2)

        st.success('✅ Done!')

        df = pd.DataFrame.from_dict(listToAppend)

        st.caption('')
        st.markdown('### Check classifier results')
        st.caption('')
        st.checkbox('Widen layout',key='widen',help="Tick this box to toggle the layout to 'Wide' mode")
        st.caption('')
        
        # df conversion
        f = [[f'{x:.2%}' for x in row] for row in df['scores']] # format score and replace
        df['classification scores'] = f
        df.drop('scores', inplace=True, axis=1)
        df.rename(columns={'sequence': 'keyphrases'}, inplace=True)

        gb = GridOptionsBuilder.from_dataframe(df)
        gb.configure_default_column(enablePiot=True, enableValue=True, enableRowGroup=True)
        gb.configure_selection(selection_mode='multiplie', use_checkbox=True)
        gb.configure_side_bar()
        gridOptions = gb.build()
        response = AgGrid(
            df,
            gridOptions=gridOptions,
            enable_enterprise_modules=True,
            update_mode=GridUpdateMode.MODEL_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            height=300,
            fit_columns_on_grid_load=False,
            configure_side_bar=True
        )

        cs, c1 = st.columns([2, 2])
        with cs:
            @st.cache
            def convert_df(df):
                return df.to_csv().encode('utf-8')
            
            csv = convert_df(df)

            st.download_button(
                label='Download results as CSV',
                data=csv,
                file_name='results.csv',
                mime='text/csv',
            )

elif selected == 'Unlocked Mode':
    with st.form(key='my_form'):
        # TODO
        API_KEY2 = st.text_input(
            'Enter your 🤗 HuggingFace API key',
            help='Once you created your HuggingFace account, you can get your free API token in you setting page: https://huggingface.co/settings/tokens'
        )
        API_URL = ('https://api-inference.huggingface.co/models/valhalla/distilbart-mnli-12-3')
        headers = {'Authorization': f'Bearer {API_KEY2}'}

        multiselectComponent = st_tags(
            label='',
            text='Add labels - 3 max',
            value=['Transactional', 'Informational', 'Navigational'],
            suggestions=[
                'Transactional', 'Informational', 'Navigational', 
                'Positive', 'Negative', 'Neutral'
            ],
            maxtags=3,
        )
        new_line = '\n'
        nums = {
            'I want to buy something in this store',
            'How to ask a question about a product',
            'Request a refund through the Google Play store',
            'I have a broken screen, what should I do?',
            'Can I have the link to the product?',
        }
        sample = f'{new_line.join(map(str, nums))}'
        linesDeduped2 = []

        MAX_LINES_FULL = 50
        text = st.text_area(
            'Enter keyphrases to classify',
            sample,
            height=200,
            key=2,
            help='At least two keyphrases for the classfier to woek, one per line, '
            + str(MAX_LINES_FULL)
            + "keyphrases max in 'Unlocked mode'. You can tweak 'MAX_LINES_FULL' in the code to change this"
        )
        lines = text.split('\n')
        linesList = []
        for x in lines:
            linesList.append(x)
        linesList = list(dict.fromkeys(linesList)) # Remove dupes
        linesList = list(filter(None, linesList)) # Remove empty

        if len(linesList) > MAX_LINES_FULL:
            st.info(
                f'❄️  Only the first '
                + str(MAX_LINES_FULL)
                + " keyprases will be reviewed to preserve performance. Fork the repo and tweak 'MAX_LINES_FULL' in the code to increase that limit."
            )
        linesList = linesList[:MAX_LINES_FULL]

        submit_button = st.form_submit_button(label='Submit')

    if not submit_button and not st.session_state.valid_inputs_received:
        st.stop()

    elif submit_button and not text:
        st.warning('❄️ There is no keyphrases to classify')
        st.session_state.value_inputs_received = False
        st.stop()

    elif submit_button and not multiselectComponent:
        st.warning('❄️ You have not added any labels, please add some! ')
        st.session_state.value_inputs_received = False
        st.stop()
    
    elif submit_button and len(multiselectComponent) == 1:
        st.warning('❄️ Please make sure to add at least two labels for classification')
        st.session_state.value_inputs_received = False
        st.stop()

    elif submit_button or st.session_state.value_inputs_received:
        try:
            if submit_button:
                st.session_state.value_inputs_received = True

            def query(payload):
                response = requests.post(API_URL, headers=headers, json=payload)
                return response.json()
            
            listtest = ['I want a refund', 'I have a question']
            listToAppend = []
            for row in linesList:
                output2 = query(
                    {
                        'inputs': row,
                        'parameters': {'candidate_labels': multiselectComponent},
                        'options': {'wait_for_model': True}
                    }
                )
                listToAppend.append(output2)
                df = pd.DataFrame.from_dict(output2)

            st.success('✅ Done!')

            df = pd.DataFrame.from_dict(listToAppend)

            st.caption('')
            st.markdown('### Check classifier results')
            st.caption('')
            st.checkbox('Widen layout',key='widen',help="Tick this box to toggle the layout to 'Wide' mode")
            st.caption('')
            
            # df conversion
            f = [[f'{x:.2%}' for x in row] for row in df['scores']] # format score and replace
            df['classification scores'] = f
            df.drop('scores', inplace=True, axis=1)
            df.rename(columns={'sequence': 'keyphrases'}, inplace=True)

            gb = GridOptionsBuilder.from_dataframe(df)
            gb.configure_default_column(enablePiot=True, enableValue=True, enableRowGroup=True)
            gb.configure_selection(selection_mode='multiplie', use_checkbox=True)
            gb.configure_side_bar()
            gridOptions = gb.build()
            response = AgGrid(
                df,
                gridOptions=gridOptions,
                enable_enterprise_modules=True,
                update_mode=GridUpdateMode.MODEL_CHANGED,
                data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
                height=300,
                fit_columns_on_grid_load=False,
                configure_side_bar=True
            )

            cs, c1 = st.columns([2, 2])
            with cs:
                @st.cache
                def convert_df(df):
                    return df.to_csv().encode('utf-8')
                
                csv = convert_df(df)

                st.caption('')
                st.download_button(
                    label='Download results as CSV',
                    data=csv,
                    file_name='results.csv',
                    mime='text/csv',
                )
        except ValueError as ve:
            st.warning('❄️ Add a valid HuggingFace API key in the text box above ☝️')
            st.stop()

if __name__ == '__main__':
    main()

keyboard_to_url(
    key='g',
    url='https://github.com/CharlyWargnier/zero-shot-classifier/blob/main/streamlit_app.py',
)

keyboard_to_url(
    key_code=190,
    url='https://github.com/CharlyWargnier/zero-shot-classifier/blob/main/streamlit_app.py',
)