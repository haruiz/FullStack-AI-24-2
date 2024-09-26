from fastapi import FastAPI
from contextlib import asynccontextmanager
from models import IrisModel, FlowersModel, Framework
from iris_model_router import router as iris_model_router
from flowers_model_router import router as flowers_model_router
from pathlib import Path


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    print("here you should add the code you want to run when the app is starting")
    print("Loading the ML models.....")
    # create a dictionary to save a ref to the registered models (model garden)
    app.state.model_garden = dict()
    # register models in the model garden
    base_model_folder = Path("./../models")
    iris_model = IrisModel(framework=Framework.SKLEARN, model_path=base_model_folder / "iris-model/sklearn/model.pk")
    flowers_model = FlowersModel(framework=Framework.TENSORFLOW, model_path= base_model_folder / "flowers-model/model.keras")

    app.state.model_garden["iris-model"] = iris_model
    app.state.model_garden["flowers-model"] = flowers_model

    yield  
    # Clean up the ML models and release the resources
    print("here you should add the code you want to run when the app is shutting down")

# creating the API
app = FastAPI(lifespan=lifespan)
app.include_router(iris_model_router, prefix="/iris-model")
app.include_router(flowers_model_router, prefix="/flowers-model")

# creating global routes
@app.get("/")
async def root():
    return {"message": "Welcome to the models API"}


