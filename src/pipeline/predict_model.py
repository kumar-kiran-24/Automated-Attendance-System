import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# Load model & class names
model = tf.keras.models.load_model("C:\ht\src\pipeline\cnn_model.h5")
class_names = ["rohit","virat"]

# Load a test image
img_path = "C:\ht\src\pipeline\download1.jpg"  # 👈 replace with your test image path
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0) / 255.0  # normalize

# Predict
predictions = model.predict(img_array)

# Show results
print("Raw output (probabilities):", predictions)

predicted_index = np.argmax(predictions, axis=1)[0]
print("Predicted class index:", predicted_index)
print("Predicted class name:", class_names[predicted_index])
