import cv2
import os
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException


class VideoToImage:

    @staticmethod
    def video_to_frames(video_path, frame_skip=1):
        """
        Convert a video into frames (images) and save in raw_frames/<video_name>/.
        """
        try:
            # Get video name without extension
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Create output folder
            output_dir = os.path.join("raw_frames", video_name)
            os.makedirs(output_dir, exist_ok=True)

            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise CustomException(f"Cannot open video file: {video_path}", sys.exc_info())

            frame_count = 0
            saved_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    img_name = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
                    cv2.imwrite(img_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_count += 1

                frame_count += 1

            cap.release()
            logging.info(f"Done! Extracted {saved_count} frames to '{output_dir}'")
            return output_dir  # return the folder path

        except Exception as e:
            raise CustomException(str(e), sys.exc_info())

    @staticmethod
    def process_folder(folder_path, frame_skip=1):
        """
        Process all videos in a folder and extract frames.
        """
        try:
            if not os.path.exists(folder_path):
                raise CustomException(f"Folder not found: {folder_path}", sys.exc_info())

            supported_exts = (".mp4", ".avi", ".mov", ".mkv")
            results = {}

            for file in os.listdir(folder_path):
                if file.lower().endswith(supported_exts):
                    video_path = os.path.join(folder_path, file)
                    logging.info(f"Processing video: {video_path}")
                    output = VideoToImage.video_to_frames(video_path, frame_skip)
                    results[file] = output

            return results

        except Exception as e:
            raise CustomException(str(e), sys.exc_info())


# Example usage
if __name__ == "__main__":
    # Example 1: Single video
    video_path = r"C:\ht\captured_data\video\2025_09_17_00_54_31.mp4"
    VideoToImage.video_to_frames(video_path, frame_skip=5)

    # Example 2: All videos in a folder
    folder_path = r"C:\ht\captured_data\video"
    VideoToImage.process_folder(folder_path, frame_skip=10)
