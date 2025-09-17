from src.utils.logger import logging
from src.utils.exception import CustomException
from src.component.capture.video_capture import VideoRecorder
from src.component.capture.image_capture import ImageCapture
from src.model.face_recognizer import FaceRecognizer

from src.pipeline.attendence_counter import AttendanceMarker
from src.data_preprocessing.image_conevrter import ImageConverter
from src.pipeline.attendence_counter import AttendanceMarker



import sys
import datetime
from dataclasses import dataclass

# your other imports here...




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
            self.image_converter = ImageConverter()
            self.attendance_counter=AttendanceMarker()
        except Exception as e:
            logging.error("Error initializing Main class")
            raise CustomException(e, sys)

    def initiate_main(self):
        try:
            # Step 1: Record video
            print("\n\nstart the recorde of the video\n\n")
            path_to_video_recorded = self.video_recorder.initiate_videorecorder()
            print("\n\nend  the recorde of the video\n\n",path_to_video_recorded)
           

            # Step 2: Convert video into frames
            print("\n\n start the converting  the video\n\n")
            path_to_selected_frames = self.image_converter.process_video(path_to_video_recorded,min_faces=2)
            print("\n\n end the converting  the video\n\n",path_to_selected_frames)


            print("\n\n start the cregonization\n\n")
            data_of_fames_in_image=self.face_recognizer.recognize_images_in_folder(path_to_selected_frames)
            print("end the regonization")

            all_students = ["Alice", "Bob", "Charlie", "David", "virat", "rohit"]

            
            absent,prsent=self.attendance_counter.initiate_mark_attendance(data_of_fames_in_image,all_students,threshold=2)
            print("presnt",prsent)
            print("absent",absent)
            return (f"absent{absent}\n prsent{prsent}")

        except Exception as e:
            logging.error("Error in initiate_main")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = Main()
    obj.initiate_main()
