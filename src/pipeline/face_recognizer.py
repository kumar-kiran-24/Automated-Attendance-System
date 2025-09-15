import os
import cv2
import numpy as np
import insightface
from numpy.linalg import norm


# Path to embeddings (already created)
EMBEDDINGS_PATH = "C:/ht/embeddings"
MODEL_PATH = "C:/ht/models"
THRESHOLD = 0.6   # similarity threshold (tune if needed)


class FaceRecognizer:
    def __init__(self):
        # Load ArcFace + RetinaFace together
        self.app = insightface.app.FaceAnalysis(
            name="buffalo_l",
            root=MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )
        self.app.prepare(ctx_id=0, det_size=(640, 640))

        # Load embeddings from disk
        self.known_embeddings = self.load_embeddings()

    def load_embeddings(self):
        people_embeddings = {}
        for person in os.listdir(EMBEDDINGS_PATH):
            person_folder = os.path.join(EMBEDDINGS_PATH, person)
            if os.path.isdir(person_folder):
                mean_path = os.path.join(person_folder, "mean_embedding.npy")
                if os.path.exists(mean_path):
                    mean_vec = np.load(mean_path)
                    people_embeddings[person] = mean_vec
        print("Loaded embeddings for:", list(people_embeddings.keys()))
        return people_embeddings

    def recognize_face(self, face_embedding):
        best_match = "Unknown"
        best_score = -1

        for person, saved_embedding in self.known_embeddings.items():
            # cosine similarity
            sim = np.dot(face_embedding, saved_embedding) / (norm(face_embedding) * norm(saved_embedding))
            if sim > best_score:
                best_score = sim
                best_match = person

        if best_score < THRESHOLD:
            return "Unknown", best_score
        return best_match, best_score

    def run_camera(self, cam_id=0):
        cap = cv2.VideoCapture(cam_id)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces
            faces = self.app.get(frame)

            for face in faces:
                bbox = face.bbox.astype(int)
                embedding = face.normed_embedding

                # Recognize face
                name, score = self.recognize_face(embedding)

                # Draw box + label
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} ({score:.2f})", (bbox[0], bbox[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Face Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    fr = FaceRecognizer()
    fr.run_camera()
