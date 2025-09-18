import os
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.svm import SVC
from mtcnn import MTCNN
from keras.models import load_model
import cv2



# Load FaceNet model

facenet_model = load_model("facenet_keras.h5")
print("[INFO] FaceNet model loaded.")


def extract_face(filename, required_size=(160, 160)):
    """Detect and extract face using MTCNN"""
    image = Image.open(filename)
    image = image.convert("RGB")
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    if len(results) == 0:
        return None
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = Image.fromarray(face)
    image = image.resize(required_size)
    return np.asarray(image)

def get_embedding(model, face_pixels):
    face_pixels = face_pixels.astype("float32")
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    sample = np.expand_dims(face_pixels, axis=0)
    yhat = model.predict(sample)
    return yhat[0]


dataset_path = "dataset"
X, y = [], []

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_folder):
        continue
    for img_name in os.listdir(person_folder):
        img_path = os.path.join(person_folder, img_name)
        face = extract_face(img_path)
        if face is not None:
            embedding = get_embedding(facenet_model, face)
            X.append(embedding)
            y.append(person_name)

X = np.asarray(X)
y = np.asarray(y)

print(f"[INFO] Loaded {X.shape[0]} face embeddings.")


# Train classifier (SVM)

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

in_encoder = Normalizer(norm="l2")
X = in_encoder.transform(X)

model_svm = SVC(kernel="linear", probability=True)
model_svm.fit(X, y_encoded)
print("[INFO] SVM classifier trained.")

# Step 3: Recognize faces in group photo

group_img = cv2.imread("group_photo.jpg")
rgb_img = cv2.cvtColor(group_img, cv2.COLOR_BGR2RGB)

detector = MTCNN()
results = detector.detect_faces(rgb_img)

for res in results:
    x1, y1, width, height = res['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    
    face = rgb_img[y1:y2, x1:x2]
    face = Image.fromarray(face).resize((160, 160))
    face = np.asarray(face)
    
    embedding = get_embedding(facenet_model, face)
    embedding = in_encoder.transform([embedding])
    
    yhat_class = model_svm.predict(embedding)
    yhat_prob = model_svm.predict_proba(embedding)
    
    class_index = yhat_class[0]
    class_probability = yhat_prob[0,class_index] * 100
    predict_name = encoder.inverse_transform(yhat_class)[0]
    
    label = f"{predict_name} ({class_probability:.2f}%)"
    
    # Draw bounding box
    cv2.rectangle(group_img, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(group_img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, (0,255,0), 2)

cv2.imshow("Face Recognition", group_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
