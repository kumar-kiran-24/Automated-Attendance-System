import cv2
import os
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException


class VideoToImage:

    @staticmethod
    def video_to_frames(video_path, frame_skip=1):
        """
        Convert video into frames (images) and save in raw_frames/<video_name>/.

        Args:
            video_path (str): Path to input video file.
            frame_skip (int): Save every Nth frame (default=1 → every frame).
        """
        try:
            # Get video name without extension
            video_name = os.path.splitext(os.path.basename(video_path))[0]

            # Create output folder structure: raw_frames/<video_name>
            output_dir = os.path.join("raw_frames", video_name)
            os.makedirs(output_dir, exist_ok=True)

            # Open video file
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
                    # Save frame as high-quality JPEG
                    img_name = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
                    cv2.imwrite(img_name, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    saved_count += 1

                frame_count += 1

            cap.release()
            logging.info(f"✅ Done! Extracted {saved_count} frames to '{output_dir}'")

        except Exception as e:
            raise CustomException(str(e), sys.exc_info())


# Example usage
if __name__ == "__main__":
    video_path = r"C:\ht\captured_data\video\2025_09_17_00_54_31.mp4"  # your video path
    VideoToImage.video_to_frames(video_path, frame_skip=5)  # saves every 5th frame
