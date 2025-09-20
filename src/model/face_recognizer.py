import os
import cv2
import numpy as np
import insightface
from numpy.linalg import norm
from dataclasses import dataclass

from src.utils.logger import logging
from src.utils.exception import CustomException
from insightface.app import FaceAnalysis


# Path to embeddings (already created)
EMBEDDINGS_PATH = "C:/ht/embeddings"
MODEL_PATH = "C:/ht/models"
THRESHOLD = 0.5   # similarity threshold


@dataclass
class FaceRecognizerConfig:
    captured_data_path: str = r"C:\ht\raw_frames\2025_09_18_12_34_22"


class FaceRecognizer:
    def __init__(self):
        self.facerecognizeconfig = FaceRecognizerConfig()

        # Load ArcFace + RetinaFace
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Load embeddings from disk
        self.known_embeddings = self.load_embeddings()

    def intiatefaceregonizer(self):
        """Reinitialize the model if needed"""
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.known_embeddings = self.load_embeddings()

    def load_embeddings(self):
        """Load stored mean embeddings for each person"""
        people_embeddings = {}
        for person in os.listdir(EMBEDDINGS_PATH):
            person_folder = os.path.join(EMBEDDINGS_PATH, person)
            if os.path.isdir(person_folder):
                mean_path = os.path.join(person_folder, "mean_embedding.npy")
                if os.path.exists(mean_path):
                    mean_vec = np.load(mean_path)
                    people_embeddings[person] = mean_vec
        print("Loaded embeddings for:", list(people_embeddings.keys()))
        logging.info(f"Loaded embeddings for persons: {list(people_embeddings.keys())}")
        return people_embeddings

    def recognize_face(self, face_embedding):
        """Compare face embedding with known embeddings"""
        logging.info("Start face recognition for detected embedding")
        best_match = "Unknown"
        best_score = -1

        for person, saved_embedding in self.known_embeddings.items():
            sim = np.dot(face_embedding, saved_embedding) / (norm(face_embedding) * norm(saved_embedding))
            if sim > best_score:
                best_score = sim
                best_match = person

        if best_score < THRESHOLD:
            return "Unknown", best_score
        return best_match, best_score

    def recognize_images_in_folder(self, folder_path, output_dir="recognized_results"):
        """Process all images in a folder and recognize faces"""
        os.makedirs(output_dir, exist_ok=True)

        results = {}  # dictionary to store all results

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            # only process images
            if not img_path.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            frame = cv2.imread(img_path)
            if frame is None:
                print(f"Could not read {img_path}")
                continue

            faces = self.app.get(frame)
            detected_names = []

            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.normed_embedding

                # Recognize face
                name, score = self.recognize_face(embedding)
                detected_names.append(f"{name}")

                # Draw results
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({score:.2f})", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Save recognized image
            save_path = os.path.join(output_dir, f"recognized_{img_name}")
            cv2.imwrite(save_path, frame)

            # Save results
            results[img_name] = detected_names
            print(f"{img_name} → {detected_names} | saved to {save_path}")

        return results 
    
     # return all results after processing folder


if __name__ == "__main__":
    fr = FaceRecognizer()
    folder = r"C:\ht\raw_frames\2025_09_18_12_34_22"   # Replace with your input folder path
    results = fr.recognize_images_in_folder(folder)
    print("\nFinal Results:", results)
    logging.info(results)
