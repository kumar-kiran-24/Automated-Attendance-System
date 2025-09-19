from src.utils.logger import logging
from src.utils.exception import CustomException
from src.component.capture.video_capture import VideoRecorder
from src.component.capture.image_capture import ImageCapture
from src.model.face_recognizer import FaceRecognizer
from src.pipeline.attendence_counter import AttendanceMarker
from src.data_preprocessing.image_selector import ImageSelector
from src.data_preprocessing.video_to_image import VideoToImage

import sys
from dataclasses import dataclass
import json
from datetime import datetime, timezone
import os

# New import
from flask import Flask, jsonify


@dataclass
class MainConfig:
    pass


class Main:
    def __init__(self):
        try:
            self.face_recognizer = FaceRecognizer()
            self.attendance_counter = AttendanceMarker()
            self.video_recorder = VideoRecorder()
            self.image_capturer = ImageCapture()
            self.video_to_image = VideoToImage()
            self.image_selector = ImageSelector()
        except Exception as e:
            logging.error("Error initializing Main class")
            raise CustomException(e, sys)

    def initiate_main(self):
        try:
            # Step 1: Record video
            recorded_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            print("\n\nStart recording the video\n\n")
            path_to_video_recorded = self.video_recorder.initiate_videorecorder()
            print("\n\nEnd recording the video\n\n", path_to_video_recorded)

            # Ensure path is string
            path_to_video_recorded = str(path_to_video_recorded)
            print("Recorded video path:", path_to_video_recorded)

            # Step 2: Convert video into frames
            print("\n\nStart converting the video to images\n\n")
            path_to_raw_frames = self.video_to_image.video_to_frames(
                path_to_video_recorded, frame_skip=5
            )
            print("Frames saved at:", path_to_raw_frames)

            # Step 3: Recognize faces
            print("\n\nStart recognizing the images in the selected path\n\n")
            students_per_image = self.face_recognizer.recognize_images_in_folder(
                folder_path=path_to_raw_frames
            )
            print("\n\nEnd recognizing the images\n\n")

            # Step 4: Mark attendance
            all_students = ["kiran", "prasana", "suhas", "manoj"]

            present, absent = self.attendance_counter.initiate_mark_attendance(
                students_per_image, all_students, threshold=2, min_conf=0.4
            )

            print("Present:", present)
            print("Absent:", absent)

            # Step 5: Save attendance JSON
            base_dir = "C:/ht/result"
            os.makedirs(base_dir, exist_ok=True)

            folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            folder_path = os.path.join(base_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            file_path = os.path.join(folder_path, "attendance.json")

            # Student USN mapping
            usn = {
                "kiran": "4ALIS024",
                "manoj": "4AL22IS400",
                "suhas": "4AL23IS059",
                "parsana": "4AL23IS042"
            }

            # Example lists
         

            recorded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # Collect results
            result = []

            for student in present:
                result.append({
                    "name": student,
                    "usn": usn.get(student, "NA"),
                    "status": "present",
                    "recorded_at": recorded_at,
                })

            for student in absent:
                result.append({
                    "name": student,
                    "usn": usn.get(student, "NA"),
                    "status": "absent",
                    "recorded_at": recorded_at,
                })

            # Save to JSON
            with open(file_path, "w") as f:
                json.dump(result, f, indent=4)

            print("Saved at:", file_path)

            return result  # ✅ Return JSON instead of string

        except Exception as e:
            logging.error("Error in initiate_main")
            raise CustomException(e, sys)


# Flask app added here
app = Flask(__name__)

@app.route("/get_attendance", methods=["GET"])
def get_attendance():
    obj = Main()
    data = obj.initiate_main()
    return jsonify(data)


if __name__ == "__main__":
    # Run Flask app
    # app.run(host="0.0.0.0", port=5000, debug=True)
    obj=Main()
    result=obj.initiate_main()
    print(result)
    
