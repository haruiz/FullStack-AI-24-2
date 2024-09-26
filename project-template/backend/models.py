import numpy as np
from enum import Enum, auto
from pathlib import Path
import typing
from abc import ABC, abstractmethod
import joblib
import tensorflow as tf
from PIL import Image as PILImage
from io import BytesIO


class Framework(Enum):
    TENSORFLOW = auto()
    SKLEARN = auto()
    PYTORCH = auto()


class Model(ABC):
    def __init__(self, 
                 model_name: str,
                 model_path: typing.Union[str, Path], 
                 framework: Framework, 
                 classes: typing.List[str]):
        """
        Abstract base model class to handle loading and predicting using different frameworks.
        """
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.framework = framework
        self.classes = classes
        self.model = None
        self.load()

    def load(self):
        """
        Load the model from the model_path based on the framework.
        Raises:
            ValueError: If model file is not found or framework is unsupported.
        """
        if not self.model_path.exists():
            raise ValueError(f"Model file {self.model_path} not found.")

        if self.framework == Framework.SKLEARN:
            self._load_sklearn_model()
        elif self.framework == Framework.TENSORFLOW:
            self._load_tensorflow_model()
        else:
            raise ValueError(f"Framework {self.framework} is not supported.")

    def _load_sklearn_model(self):
        """
        Load a sklearn model using joblib.
        """
        self.model = joblib.load(self.model_path)

    def _load_tensorflow_model(self):
        """
        Load a TensorFlow model using Keras.
        """
        self.model = tf.keras.models.load_model(self.model_path)

    @abstractmethod
    def predict(self, X: typing.Any) -> typing.Any:
        """
        Make a prediction on the input data.
        """
        pass

    def __call__(self, X: typing.Any) -> typing.Any:
        """
        Allow the model instance to be callable, directing to predict method.
        """
        return self.predict(X)


class IrisModel(Model):
    def __init__(self, framework: Framework = Framework.TENSORFLOW, model_path: typing.Union[str, Path] = None):
        """
        Initialize an Iris model, supporting both TensorFlow and sklearn frameworks.
        """
        classes = ["setosa", "versicolor", "virginica"]
        super().__init__("iris-model", model_path, framework, classes)

    def predict(self, X: np.ndarray) -> typing.List[dict]:
        """
        Make a prediction using the loaded Iris model.
        Args:
            X: Input data, numpy array (for sklearn) or tf.Tensor (for TensorFlow).
        Returns:
            List of dictionaries with class probabilities.
        """
        if self.framework == Framework.SKLEARN:
            predictions = self.model.predict_proba(X)
        elif self.framework == Framework.TENSORFLOW:
            predictions = self.model.predict(X)
        else:
            raise ValueError(f"Framework {self.framework} not supported for prediction.")
        
        outputs = [
            {self.classes[j]: round(float(prob), 3) for j, prob in enumerate(xi_probs)}
            for xi_probs in predictions
        ]
        return outputs


class FlowersModel(Model):
    def __init__(self, framework: Framework = Framework.TENSORFLOW, model_path: typing.Union[str, Path] = None):
        """
        Initialize a TensorFlow-based Flowers model.
        """
        classes = ["daisy", "dandelion", "roses", "sunflowers", "tulips"]
        super().__init__("flowers-model", model_path, Framework.TENSORFLOW, classes)
        self.target_size = (180, 180)

    def _preprocess_image(self, image_bytes: bytes) -> tf.Tensor:
        """
        Preprocess the input image for the TensorFlow model.
        Args:
            image_bytes: Raw image data in bytes.
        Returns:
            Tensor of the preprocessed image, normalized and resized.
        """
        image_stream = BytesIO(image_bytes)
        image = PILImage.open(image_stream)
        image = image.resize(self.target_size)
        image_arr = np.array(image)
        image_tensor = tf.convert_to_tensor(image_arr, dtype=tf.float32)
        image_tensor = tf.expand_dims(image_tensor, 0)  # Add batch dimension
        image_tensor = image_tensor / 255.0  # Normalize to [0,1]
        return image_tensor

    def predict(self, image_bytes: bytes) -> typing.List[dict]:
        """
        Make a prediction on the input image.
        Args:
            image_bytes: Raw image data in bytes.
        Returns:
            List of dictionaries with class probabilities.
        """
        img_tensor = self._preprocess_image(image_bytes)
        scores = self.model.predict(img_tensor)
        predictions = tf.nn.softmax(scores)
        
        outputs = [
            {self.classes[j]: round(float(prob), 3) for j, prob in enumerate(xi_probs)}
            for xi_probs in predictions
        ]
        return outputs
