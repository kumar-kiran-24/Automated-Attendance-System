import os
import sys
import numpy as np
import insightface
from PIL import Image
from dataclasses import dataclass

from src.utils.exception import CustomException
from src.utils.logger import logging


@dataclass
class FaecEmbeddingConfig:
    train_path: str = os.path.join("data", "train")
    val_path: str = os.path.join("data", "val")
    output_path: str = os.path.join("C:/ht/embeddings")   # <--- embeddings will be stored here


class FaecEmbedding:
    def __init__(self):
        self.config = FaecEmbeddingConfig()
        os.makedirs(self.config.output_path, exist_ok=True)

    def initae_faec_embedding(self):
        logging.info("Face embedding started")
        try:
            # Initialize ArcFace model
            app = insightface.app.FaceAnalysis(
                name='buffalo_l',
                root='C:/ht/models',
                providers=['CPUExecutionProvider']
            )
            app.prepare(ctx_id=0)

            # Extract embedding from a single image
            def get_embedding(image_path):
                img = np.array(Image.open(image_path).convert("RGB"))
                faces = app.get(img=img)

                if len(faces) == 0:
                    return None

                # Pick the largest face
                faces = sorted(faces, key=lambda x: x.bbox[2] - x.bbox[0], reverse=True)
                embedding = faces[0].normed_embedding  # 512-D vector
                return embedding

            # Process all images inside one folder (person)
            def folder_to_embeddings(person_name, folder_path):
                embeddings = []
                for file in os.listdir(folder_path):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        emd = get_embedding(os.path.join(folder_path, file))
                        if emd is not None:
                            embeddings.append(emd)

                embeddings = np.array(embeddings)
                if len(embeddings) > 0:
                    mean_embedding = np.mean(embeddings, axis=0)

                    # Save embeddings to disk
                    person_dir = os.path.join(self.config.output_path, person_name)
                    os.makedirs(person_dir, exist_ok=True)

                    np.save(os.path.join(person_dir, "all_embeddings.npy"), embeddings)
                    np.save(os.path.join(person_dir, "mean_embedding.npy"), mean_embedding)

                    return embeddings, mean_embedding
                else:
                    return None, None

            # Process the whole dataset (all persons)
            def dataset_to_embeddings(dataset_path):
                people_embeddings = {}
                for person in os.listdir(dataset_path):
                    person_folder = os.path.join(dataset_path, person)
                    if os.path.isdir(person_folder):
                        all_vecs, mean_vec = folder_to_embeddings(person, person_folder)
                        if mean_vec is not None:
                            people_embeddings[person] = {
                                "all_embeddings": all_vecs,
                                "mean_embedding": mean_vec
                            }
                return people_embeddings

            # Run on training dataset
            dataset_path = self.config.train_path
            people_embeddings = dataset_to_embeddings(dataset_path)
            logging.info("face embedding is done")

            # Print results
            for person, data in people_embeddings.items():
                print(f"\nPerson: {person}")
                print(" - All embeddings shape:", data["all_embeddings"].shape)
                print(" - Mean embedding shape:", data["mean_embedding"].shape)
                print(f" - Saved at: {os.path.join(self.config.output_path, person)}")

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = FaecEmbedding()
    obj.initae_faec_embedding()
