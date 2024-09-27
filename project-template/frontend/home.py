import streamlit as st
from client import call_iris_model, call_flowers_model
from requests.exceptions import ConnectionError

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
        try:
            with st.spinner('Predicting...'):
                response = call_iris_model(sepal_length, sepal_width, petal_length, petal_width)
                
            if response.status_code != 200:
                st.error(f"Error: {response.text}")
            json_result = response.json()
            st.write(json_result)
            st.snow()
        except ConnectionError as ex:
            st.error(f"Connection error: {ex}")
        except Exception as ex:
            st.error(f"Uknown error : {ex}")

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
            try:
                with st.spinner('Predicting...'):
                    response = call_flowers_model(image_file)
                    
                if response.status_code != 200:
                    st.error(f"Error: {response.text}")
                json_result = response.json()
                st.write(json_result)
                st.snow()
            except ConnectionError as ex:
                st.error(f"Connection error: {ex}")
            except Exception as ex:
                st.error(f"Uknown error : {ex}")


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