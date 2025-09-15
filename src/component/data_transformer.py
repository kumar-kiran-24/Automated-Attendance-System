import os
import sys
from dataclasses import dataclass
import tensorflow as tf

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataTransformerConfig:
    train_path: str = os.path.join("data", "train")
    val_path: str = os.path.join("data", "val")
    batch_size: int = 32


class DataTransformer:
    def __init__(self):
        self.config = DataTransformerConfig()

    def initiate_data_transformation(self):
        logging.info("Data transformation started")
        try:
            img_size = (224, 224)
            batch_size = self.config.batch_size

            # Load train data
            train_data = tf.keras.utils.image_dataset_from_directory(
                self.config.train_path,
                image_size=img_size,
                batch_size=batch_size,
                label_mode="categorical"
            )

            # Load val data
            val_data = tf.keras.utils.image_dataset_from_directory(
                self.config.val_path,
                image_size=img_size,
                batch_size=batch_size,
                label_mode="categorical"
            )

            class_names = train_data.class_names
            logging.info(f"Classes found: {class_names}")

            # Normalize images (0–1 range)
            normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)
            train_data = train_data.map(lambda x, y: (normalization_layer(x), y))
            val_data = val_data.map(lambda x, y: (normalization_layer(x), y))

            logging.info("Data transformation completed successfully")
            return train_data, val_data, class_names

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformer()
    train_loader, val_loader, class_names = obj.initiate_data_transformation()
    logging.info(f"Classes: {class_names}")
    logging.info(f"Train batches: {len(list(train_loader))}, Val batches: {len(list(val_loader))}")
