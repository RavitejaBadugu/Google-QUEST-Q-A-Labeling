import streamlit as st
import requests
import pandas as pd
from test_params import streamlit_settings

fastapi=streamlit_settings.fastapi_url
send_data=streamlit_settings.send_data_url
get_data=streamlit_settings.get_data_url

st.header("streamlit app for google quest labelling competition")
st.subheader("Enter the data to get predictions and if you want can also check the examples of input given to  the model below")

question=st.text_area(label="Write the question here")
answer=st.text_area(label="Write the answer here")
title=st.text_area(label="Write the title here")

n=st.number_input('enter the number of examples to get')
if n>0:
    sample_data=requests.post(get_data,json={'n':n}).json()
    st.dataframe(pd.DataFrame(data=sample_data,columns=['id','questions','answers','title','posted_date']))
        
else:
    st.error('number must be greater then 0')
if st.button('click to get predictions'):
    if (question!='') and (title!='') and (answer!=''):
        with st.spinner('its in progress'):
            score=requests.post(send_data,json={'TITLE':title,'QUESTION':question,
                                    'ANSWER':answer})
            st.success('models predicted')
            st.json(score.json())
    else:
        if question=='':
            st.error('please enter question')
        elif answer=='':
            st.error('please enter answer')
        else:
            st.error('please enter title')