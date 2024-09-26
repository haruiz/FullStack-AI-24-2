import streamlit as st
import requests
import json
import os

API_URL = os.getenv('API_URL', 'http://localhost:8080')
print("API_URL", API_URL)


def call_iris_model(sepal_length, sepal_width, petal_length, petal_width):
    """
    This function calls the iris model
    """
    url = f"{API_URL}/iris-model/predict"

    payload = json.dumps({
    "sepal_length": sepal_length,
    "sepal_width": sepal_width,
    "petal_length": petal_length,
    "petal_width": petal_width
    })
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()

def call_flowers_model(image_file):
    """
    This function calls the flowers model
    """
    url = f"{API_URL}/flowers-model/predict"
    payload = {}
    files=[
        ('image',image_file)
    ]
    headers = {}
    response = requests.request("POST", url, headers=headers, data=payload, files=files)
    return response.json()


def create_iris_model_form():
    st.header('Iris Model')

    sepal_length = st.number_input('Sepal length',
                                   min_value=0.0, 
                                   max_value=10.0, 
                                   value=10.0)
    sepal_width = st.number_input('Sepal width',
                                   min_value=0.0, 
                                   max_value=10.0, 
                                   value=5.0)
    petal_length = st.number_input('Petal length', 
                                   min_value=0.0, 
                                   max_value=10.0, 
                                   value=5.0)
    petal_width = st.number_input('Petal width', 
                                  min_value=0.0, 
                                  max_value=10.0, 
                                  value=5.0)
    
    is_clicked = st.button('Predict')
    if is_clicked:
        with st.spinner('Predicting...'):
            json_result = call_iris_model(sepal_length, sepal_width, petal_length, petal_width)
        st.write(json_result)
        st.balloons()

def create_flowers_model_form():
    st.header('Flowers Model')
    image_file = st.file_uploader(
        "Upload an image", 
        type=["jpg", "png", "jpeg"]
    )
    if image_file is not None:
        st.image(image_file, 
                 caption='Uploaded Image.',
                 width=200,
                 use_column_width=False)
        is_clicked = st.button('Predict')
        if is_clicked:
            with st.spinner('Predicting...'):
                json_result = call_flowers_model(image_file)
            st.write(json_result)
            st.snow()

def app():
    st.set_page_config(
        page_title='Home Page', 
        page_icon='🌍', 
        layout='centered', 
        initial_sidebar_state='auto'
    )

    st.title('Welcome to the my amazing app')
    st.write('This is a simple example of some text in the home page')
    
    option_selected = st.selectbox(
        'Select the Model', 
        ['Iris Model', 'Flowers Model']
    )

    if option_selected == 'Iris Model':
        create_iris_model_form()
    else:
        create_flowers_model_form()
    
      
if __name__ == '__main__':
    app()