import streamlit as st
import pandas as pd

st.title('Home')
st.write('Welcome to the home page!')

data = pd.DataFrame({
    'x': [1, 2, 3, 4, 5],
    'y': [10, 20, 30, 40, 50]
})

st.file_uploader('Upload file')
is_pressed = st.button('Click me')
if is_pressed:
    st.balloons()
    st.write(data)

st.text_input('Enter your name')
st.text_area('Enter your message')
