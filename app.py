from flask import Flask, request, jsonify, render_template
import os
import threading
from werkzeug.utils import secure_filename
from flask_cors import CORS   #Import CORS
from src.model.face_embedding import FaecEmbedding

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})   

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def train_student_model_async(usn, folder_path):
    try:
        faece_embedding = FaecEmbedding()
        faece_embedding.initae_faec_embedding()
        print(f"Model training completed for {usn}")
    except Exception as e:
        print(f"Error training model for {usn}: {e}")


@app.route("/upload", methods=["GET"])
def upload_form():
    return render_template("upload.html")


@app.route("/upload_photos", methods=["POST"])
def upload_photos():
    usn = request.form.get("usn")
    if not usn:
        return jsonify({"status": "error", "message": "USN is required"}), 400

    files = request.files.getlist("images")
    if not files or len(files) == 0:
        return jsonify({"status": "error", "message": "No images uploaded"}), 400

    
    save_dir = os.path.join(r"C:\ht\data\students", secure_filename(usn))
    os.makedirs(save_dir, exist_ok=True)

    saved_count = 0
    for i, file in enumerate(files):
        if file and allowed_file(file.filename):
            filename = secure_filename(f"image_{i+1}.jpg")
            file.save(os.path.join(save_dir, filename))   # <-- Corrected line
            saved_count += 1

    if saved_count == 0:
        return jsonify({"status": "error", "message": "No valid image files uploaded"}), 400

    
    threading.Thread(target=train_student_model_async, args=(usn, save_dir)).start()

    return jsonify({
        "status": "success",
        "message": f"{saved_count} images saved for {usn}, model training started."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
