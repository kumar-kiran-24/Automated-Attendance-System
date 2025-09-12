import cv2
import time
import os 
import sys
import datetime
from dataclasses import dataclass

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.logger import logging
from src.exception import CustomException

@dataclass
class VideoRecoredrConfig:
      stored_path: str = os.path.join("data", "video")


class VideoRecorder:
    def __init__(self):
            
        self.videorecordconfig =VideoRecoredrConfig()
        os.makedirs(self.videorecordconfig.stored_path, exist_ok=True)
    
    def initiate_videorecorder(self):
        logging.info("started the recording of the video")
        try:
            duration=300 #seconds of video recording
            camera_index=0

            cap=cv2.VideoCapture(camera_index)

            if not cap.isOpened():
                raise CustomException("Unable to open the camers")
            
            frame_width=int(cap.get(3))
            frame_hieght=int(cap.get(4))

            filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".avi"
            file_path = os.path.join(self.videorecordconfig.stored_path, filename)

            out=cv2.VideoWriter(
                file_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                20.0,
                (frame_width,frame_hieght)
            )
            logging.info(f"Recording started  savimg in the {file_path}")

            start_time=datetime.datetime.now()
            while(datetime.datetime.now()-start_time).seconds<duration:
                ret,frame=cap.read()
                if not ret:
                    break
                out.write(frame)

                cv2.imshow("recording",frame)

            cap.release()
            out.release()
            cv2.destroyAllWindows()
            logging.info("Recording finished and saved.")


            


        except Exception as e:
            raise CustomException(e,sys)
        
# if __name__=="__main__":
#     vc=VideoRecorder()
#     vc.initiate_videorecorder()
            


