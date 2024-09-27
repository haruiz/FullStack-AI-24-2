from fastapi import FastAPI, Request, UploadFile, File
from contextlib import asynccontextmanager
from models import IrisModel, FlowersModel, Framework
from iris_model_router import router as iris_model_router
from flowers_model_router import router as flowers_model_router
from pathlib import Path
from fastapi.responses import JSONResponse
from datetime import datetime
import numpy as np


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model_garden = dict()
    print("api starting...")
    app.state.model_garden["iris"] = IrisModel(
        framework=Framework.SKLEARN,
        model_path="./../models/iris-model/sklearn/model.pk",
    )
    app.state.model_garden["flowers"] = FlowersModel(
        framework=Framework.TENSORFLOW,
        model_path="./../models/flowers-model/model.keras",
    )
    yield
    print("api shutting down...")


app = FastAPI(lifespan=lifespan)


@app.get("/")
def say_hi():
    return JSONResponse(content={"message": "hi!"}, status_code=200)


@app.post("/iris-model/predict")
async def predict_iris(request: Request):
    iris_model = request.app.state.model_garden["iris"]
    data = await request.json()
    sepallength = data["sepal_length"]
    sepalwidth = data["sepal_width"]
    petallength = data["petal_length"]
    petalwidth = data["petal_width"]
    prediction = iris_model.predict(
        np.array([[sepallength, sepalwidth, petallength, petalwidth]])
    )
    return JSONResponse(content={"prediction": prediction}, status_code=200)


@app.post("/flowers-model/predict")
async def predict_flowers(request: Request, image: UploadFile = File(...)):
    flowers_model = request.app.state.model_garden["flowers"]
    image_bytes: bytes = await image.read()  # read the image as bytes
    predictions = flowers_model.predict(image_bytes)
    return JSONResponse(content={"predictions": predictions}, status_code=200)


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Load the ML model
#     print("here you should add the code you want to run when the app is starting")
#     print("Loading the ML models.....")
#     # create a dictionary to save a ref to the registered models (model garden)
#     app.state.model_garden = dict()
#     # register models in the model garden
#     base_model_folder = Path("./../models")
#     iris_model = IrisModel(framework=Framework.SKLEARN, model_path=base_model_folder / "iris-model/sklearn/model.pk")
#     flowers_model = FlowersModel(framework=Framework.TENSORFLOW, model_path= base_model_folder / "flowers-model/model.keras")

#     app.state.model_garden["iris-model"] = iris_model
#     app.state.model_garden["flowers-model"] = flowers_model

#     yield
#     # Clean up the ML models and release the resources
#     print("here you should add the code you want to run when the app is shutting down")

# # creating the API
# app = FastAPI(lifespan=lifespan)
# app.include_router(iris_model_router, prefix="/iris-model")
# app.include_router(flowers_model_router, prefix="/flowers-model")

# # creating global routes
# @app.get("/")
# async def root():
#     return {"message": "Welcome to the models API"}
