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
            print("\n\nStart recording the video\n\n")
            path_to_video_recorded = self.video_recorder.initiate_videorecorder()
            print("\n\nEnd recording the video\n\n", path_to_video_recorded)

            # Ensure path is string
            path_to_video_recorded = str(path_to_video_recorded)
            print("Recorded video path:", path_to_video_recorded)

            # Step 2: Convert video into frames
            print("\n\nStart converting the video to images\n\n")
            # Call as instance method without keywords
            path_to_raw_frames = self.video_to_image.video_to_frames(path_to_video_recorded, frame_skip=5)
            print("Frames saved at:", path_to_raw_frames)

            # Step 3: Select images with faces
            # Remove frame_skip because ImageSelector.main() does not accept it
            path_to_selected_images = self.image_selector.main(path=path_to_raw_frames)
            print("Selected images path:", path_to_selected_images)

            print("/\n\n\\n start the regonize the images in the selcted path\n\n\n")
            students_per_image=self.face_recognizer.recognize_images_in_folder(folder_path=path_to_selected_images)
            print("/\n\n\\n start the regonize the images in the selcted path\n\n\n")


            all_students = ["Alice", "Bob", "Charlie", "David", "virat", "rohit"]


            present,absent=self.attendance_counter.initiate_mark_attendance(
        students_per_image, all_students, threshold=2, min_conf=0.6)
    

            # Step 4: Mark attendance
            

            
            print("Present:", present)
            print("Absent:", absent)

            return f"Absent: {absent}\nPresent: {present}"

        except Exception as e:
            logging.error("Error in initiate_main")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = Main()
    obj.initiate_main()
