from fastapi import APIRouter, Request, status
from fastapi.responses import JSONResponse
from fastapi_restful.cbv import cbv
from pydantic import BaseModel

router = APIRouter()


# Model entry schema
class IrisModel(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

    def to_list(self):
        return [
            self.sepal_length,
            self.sepal_width,
            self.petal_length,
            self.petal_width,
        ]


@cbv(router)
class IrisModelCbv:
    @router.get("/hi")
    def say_hi(self):
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Hello from the Iris Model!"},
        )

    @router.post("/predict")
    async def predict(self, request: Request, data: IrisModel):
        model_input = data.to_list()
        model_garden = request.app.state.model_garden
        model = model_garden["iris-model"]
        predictions = model.predict([model_input])
        return JSONResponse(content={"prediction": predictions})
