from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from fastapi_restful.cbv import cbv
from fastapi import File, UploadFile, Form
import typing

router = APIRouter()


@cbv(router)
class FlowersModelCbv:
    @router.get("/hi")
    def say_hi(self):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Hello from the Flowers Model!"},
        )

    @router.post("/predict")
    async def predict(self, request: Request, image: UploadFile = File(...)):
        image_bytes: bytes = await image.read()  # read the image as bytes
        model_garden = request.app.state.model_garden
        flowers_model = model_garden["flowers-model"]
        predictions = flowers_model.predict(image_bytes)
        return JSONResponse(content={"predictions": predictions})
