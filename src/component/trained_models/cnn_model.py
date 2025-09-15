import sys
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformer import DataTransformer


@dataclass
class CNNModelConfig:
    datatransformer = DataTransformer()


class CNNModel:
    def __init__(self):
        self.cnn_model_config = CNNModelConfig()

    def initiate_cnn(self):
        try:
            logging.info("CNN model training started")
            datatransformer = self.cnn_model_config.datatransformer
            train_data, val_data, class_names = datatransformer.initiate_data_transformation()

            num_classes = len(class_names)

            model = Sequential([
                Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                MaxPooling2D((2, 2)),
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D((2, 2)),
                Flatten(),
                Dense(512, activation='relu'),
                Dropout(0.5),
                Dense(num_classes, activation='softmax')
            ])

            model.compile(
                optimizer='adam',
                loss='categorical_crossentropy', 
            )

            history = model.fit(
                train_data,
                validation_data=val_data,
                epochs=10
            )

            model.save("cnn_model.h5")
            logging.info("CNN model training completed and model saved as cnn_model.h5")
            return model

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    obj = CNNModel()
    obj.initiate_cnn()
    



    
