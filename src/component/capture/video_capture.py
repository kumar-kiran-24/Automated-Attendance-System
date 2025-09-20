import cv2
import os
import sys
import datetime
from dataclasses import dataclass

# add project root to sys.path so src can be imported when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.logger import logging
from src.utils.exception import CustomException


@dataclass
class VideoRecorderConfig:
    """Configuration for storing recorded videos."""
    stored_path: str = os.path.join("captured_data", "video")


class VideoRecorder:
    def __init__(self) -> None:
        self.videorecordconfig = VideoRecorderConfig()
        os.makedirs(self.videorecordconfig.stored_path, exist_ok=True)

    def initiate_videorecorder(
        self,
        duration: int = 5,
        camera_index: int = 0,
        show_preview: bool = False,
    ) -> str:
        """
        Records a video from the webcam.

        Args:
            duration (int): Recording duration in seconds (default=5).
            camera_index (int): Index of the camera to use (default=0).
            show_preview (bool): If True, show live preview window.

        Returns:
            str: Path to the saved video file.
        """
        logging.info("Started the recording of the video")
        try:
            cap = cv2.VideoCapture(camera_index)

            if not cap.isOpened():
                raise CustomException("Unable to open the camera")

            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Save as MP4 instead of AVI
            filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S") + ".mp4"
            file_path = os.path.join(self.videorecordconfig.stored_path, filename)

            # Use MP4 codec
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(
                file_path,
                fourcc,
                20.0,
                (frame_width, frame_height),
            )
            logging.info(f"Recording started, saving in {file_path}")

            start_time = datetime.datetime.now()
            while (datetime.datetime.now() - start_time).seconds < duration:
                ret, frame = cap.read()
                if not ret:
                    break
                out.write(frame)

                # Always show recording window
                cv2.imshow("Recording", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):  # press 'q' to stop early
                    break

            cap.release()
            out.release()
            cv2.destroyAllWindows()

            logging.info("Recording finished and saved.")
            return file_path

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    vr = VideoRecorder()
    saved_file = vr.initiate_videorecorder(duration=5, show_preview=True)
    print("Saved video:", saved_file)
