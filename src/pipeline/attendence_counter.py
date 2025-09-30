from src.model.face_recognizer import FaceRecognizer
from src.utils.exception import CustomException
from src.utils.logger import logging

from dataclasses import dataclass
from collections import defaultdict
import sys


@dataclass
class AttendanceMarkerConfig:
    facerecognizer = FaceRecognizer()


class AttendanceMarker:
    def __init__(self):
        self.facerecognizer = AttendanceMarkerConfig.facerecognizer

    def initiate_mark_attendance(
        self, recognizer_results: dict, all_students, threshold=2, min_conf=0.6
    ):
        """
        recognizer_results: Dict -> {image_name: [recognized students]}
        all_students: List of all enrolled students
        threshold: Minimum number of images required to mark present
        min_conf: Minimum confidence score to consider recognition valid
        """
        try:
            logging.info("Mark attendance process started")
            count_dict = defaultdict(int)

            # Loop through dictionary: image_name recognized_list
            for image_name, recognized_list in recognizer_results.items():
                for student in set(recognized_list):  # avoid duplicate count per image
                    
                    parts = student.split(" (")
                    student_name = parts[0].strip()
                    conf = (
                        float(parts[1].replace(")", "")) if len(parts) > 1 else 1.0
                    )

                    # Skip unknowns 
                    if student_name != "Unknown" and conf >= min_conf:
                        count_dict[student_name] += 1

            print("DEBUG: Count Dict =", dict(count_dict))  # Debug 

            present_students = []
            absent_students = []

            for student in all_students:
                if count_dict[student] >= threshold:
                    present_students.append(student)
                else:
                    absent_students.append(student)

            logging.info(f"Present Students: {present_students}")
            logging.info(f"Absent Students: {absent_students}")

            return present_students, absent_students

        except Exception as e:
            logging.error(f"Error in attendance marking: {str(e)}")
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = AttendanceMarker()

    # Get results from your recognizer
    recognizer_results = obj.facerecognizer.recognize_images_in_folder(
        folder_path=r"C:\ht\outputs\2025_09_18_12_34_22"  
    )

    # Define all students
    all_students = ["kiran","manoj","prasana","virat", "rohit","suhas"]

    # Run attendance
    present, absent = obj.initiate_mark_attendance(
        recognizer_results, all_students, threshold=2, min_conf=0.6
    )

    print("Present:", present)
    print("Absent:", absent)
    logging.info("prsent",present)
    logging.info("absent",absent)