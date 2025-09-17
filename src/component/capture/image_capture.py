import os
import sys
import cv2
import time
from dataclasses import dataclass
from datetime import datetime

from src.utils.exception import CustomException
from src.utils.logger import logging


@dataclass
class ImageCaptureConfig:
    stored_path=os.path.join("captured_data","Photos")

class ImageCapture:
    def __init__(self):
        self.imgcapcon=ImageCaptureConfig()
        os.makedirs(self.imgcapcon.stored_path,exist_ok=True)
    
    def intiateImageCapture(self):

        base_dir=self.imgcapcon.stored_path
        num_images=5
        delay=0.5

        timestamp=datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_dir = os.path.join(base_dir, timestamp)
        os.makedirs(output_dir, exist_ok=True)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise CustomException("unable to open the camera")
        
        logging.info("Start the capture the video")

        count=0
        zoom_factors = [1.0, 1.2, 1.5]   # Normal, zoom 1.2x, zoom 1.5x
        angles = [-20, -10, 0, 10, 20]   # Rotate face left/right


        def apply_zoom(frame, zoom_factor):
            h, w = frame.shape[:2]
            # Scale image
            new_w, new_h = int(w * zoom_factor), int(h * zoom_factor)
            resized = cv2.resize(frame, (new_w, new_h))

            # Crop back to original size (center crop)
            x1 = (new_w - w) // 2
            y1 = (new_h - h) // 2
            zoomed = resized[y1:y1+h, x1:x1+w]
            return zoomed


        while count < num_images:
            ret, frame = cap.read()
            if not ret:
                break

            zoom = zoom_factors[count % len(zoom_factors)]
            angle = angles[count % len(angles)]

            # Apply real zoom
            zoomed_frame = apply_zoom(frame, zoom)

            # Apply rotation
            h, w = zoomed_frame.shape[:2]
            M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
            rotated_frame = cv2.warpAffine(zoomed_frame, M, (w, h))

            # Save image
            img_path = os.path.join(output_dir, f"img_{count:03d}.jpg")
            cv2.imwrite(img_path, rotated_frame)
            count += 1

            cv2.imshow("Capturing Images", rotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            time.sleep(delay)

            time.sleep(delay)
        cap.release()
        cv2.destroyAllWindows()
        

        

        logging.info(f"photos capture is finish and stored in {output_dir}")
        return output_dir


if __name__=="__main__":
    Ic=ImageCapture()
    Ic.intiateImageCapture()





            


    
