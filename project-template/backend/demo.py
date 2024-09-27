from models import IrisModel, FlowersModel, Framework
import numpy as np


def main():
    iris = IrisModel(
        framework=Framework.TENSORFLOW,
        model_path="./../models/iris-model/keras/model.keras",
    )
    predictions = iris.predict(np.array([[5.1, 3.5, 1.4, 0.2]]))
    print(predictions)

    flowers = FlowersModel(
        framework=Framework.TENSORFLOW,
        model_path="./../models/flowers-model/model.keras",
    )
    with open("/Users/haruiz/Downloads/sunflowers.jpeg", "rb") as f:
        image = f.read()
    predictions = flowers.predict(image)
    print(predictions)


if __name__ == "__main__":
    main()
