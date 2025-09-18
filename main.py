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

            # Step 3: (Optional) Select images with faces
            # If ImageSelector is required, uncomment and adjust
            # path_to_selected_images = self.image_selector.extract_faces_from_folder(
            #     input_folder=path_to_raw_frames,
            #     output_base="C:/ht/selected_images"
            # )
            # print("Selected images path:", path_to_selected_images)

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

            # Step 5: Save attendance
            base_dir = "C:/ht/result"
            os.makedirs(base_dir, exist_ok=True)

            folder_name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            folder_path = os.path.join(base_dir, folder_name)
            os.makedirs(folder_path, exist_ok=True)

            file_path = os.path.join(folder_path, "attendance.json")

            # Save structured JSON: each student with status + recorded_at
            result = {}
            for student in present:
                result[student] = {
                    "usn": "NA",
                    "status": "present",
                    "recorded_at": recorded_at,
                }
            for student in absent:
                result[student] = {
                    "usn": "NA",
                    "status": "absent",
                    "recorded_at": recorded_at,
                }

            with open(file_path, "w") as f:
                json.dump(result, f, indent=4)

            print("Saved at:", file_path)

            return f"Absent: {absent}\nPresent: {present}"

        except Exception as e:
            logging.error("Error in initiate_main")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = Main()
    obj.initiate_main()
