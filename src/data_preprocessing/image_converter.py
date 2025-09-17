import os
import re
import cv2
import numpy as np
from retinaface import RetinaFace
from dataclasses import dataclass
import argparse

from src.utils.exception import CustomException
from src.utils.logger import logging


@dataclass
class ImageConverterConfig:
    base_output_dir: str = "outputs"


class ImageConverter:

    def __init__(self, config: ImageConverterConfig = ImageConverterConfig()):
        self.config = config

        # ✅ Preload RetinaFace model once (warm-up)
        logging.info("Loading RetinaFace model... (this may take some time)")
        try:
            dummy_img = np.zeros((100, 100, 3), dtype=np.uint8)
            _ = RetinaFace.detect_faces(dummy_img)
            logging.info("RetinaFace model loaded successfully ✅")
        except Exception as e:
            logging.error(f"Failed to load RetinaFace model: {e}")
            raise

    @staticmethod
    def get_safe_filename(name: str) -> str:
        """Sanitize filenames/folders for Windows."""
        return re.sub(r'[^A-Za-z0-9._-]', '_', name)

    def create_output_folders(self, video_path: str):
        """Create output folders for the given video."""
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        video_name = self.get_safe_filename(video_name)

        base_output = os.path.join(self.config.base_output_dir, video_name)
        raw_dir = os.path.join(base_output, "raw_frames")
        selected_dir = os.path.join(base_output, "selected_frames")

        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(selected_dir, exist_ok=True)

        return raw_dir, selected_dir

    @staticmethod
    def detect_faces(frame) -> int:
        """Detect faces using RetinaFace, safely."""
        try:
            detections = RetinaFace.detect_faces(frame)
            return len(detections) if isinstance(detections, dict) else 0
        except Exception:
            return 0  # fallback if RetinaFace fails

    @staticmethod
    def save_frame(frame, path: str):
        """Save a single frame to disk."""
        cv2.imwrite(path, frame)

    def process_video(self, video_path: str, min_faces: int = 1):
        """Main function to process video frames."""
        raw_dir, selected_dir = self.create_output_folders(video_path)

        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_selected = 0
        max_faces = 0

        logging.info(f"Starting video processing: {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Save raw frame
            raw_path = os.path.join(raw_dir, f"frame_{frame_count}.jpg")
            self.save_frame(frame, raw_path)

            # Detect faces
            num_faces = self.detect_faces(frame)

            # Save "selected" frame if it meets criteria
            if num_faces >= min_faces and num_faces >= max_faces:
                max_faces = num_faces
                sel_path = os.path.join(
                    selected_dir, f"frame_{frame_count}_faces_{num_faces}.jpg"
                )
                self.save_frame(frame, sel_path)
                saved_selected += 1
                logging.info(f"[SAVED] Frame {frame_count} with {num_faces} faces → {sel_path}")

            # Progress log every 50 frames
            if frame_count % 50 == 0:
                logging.info(f"Processed {frame_count} frames...")

        cap.release()
        cv2.destroyAllWindows()

        logging.info(
            f"\nProcess completed!\n"
            f"Raw frames: {raw_dir}\n"
            f"Selected frames: {selected_dir}\n"
            f"Total selected: {saved_selected}"
        )

        return selected_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video into frames and select frames with most faces.")
    parser.add_argument("video", help="Path to input video file")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Base output directory (default: outputs)")
    parser.add_argument("--min_faces", type=int, default=1, help="Minimum number of faces required to save a frame")

    args = parser.parse_args()

    try:
        config = ImageConverterConfig(base_output_dir=args.output_dir)
        converter = ImageConverter(config)
        converter.process_video(args.video, min_faces=args.min_faces)
    except Exception as e:
        import sys
        raise CustomException(e, sys)
