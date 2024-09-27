from dotenv import load_dotenv, find_dotenv
import requests
import json
import os

load_dotenv(find_dotenv())

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
    return response

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
    return response
