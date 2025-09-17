import os
import cv2
from retinaface import RetinaFace
from pathlib import Path

from src.utils.logger import logging
from src.utils.exception import CustomException


class ImageSelector:
    @staticmethod
    def extract_faces_from_folder(input_folder, output_base, max_faces=None):
        """
        Detect faces in all images from input_folder using RetinaFace 
        and save them in output_base/input_folder_name.
        
        Args:
            input_folder (str): Path to folder containing images.
            output_base (str): Base path where results will be saved.
            max_faces (int, optional): Maximum number of faces to save per image. Default = None (all faces).
        """
        logging.info("Initialize the image selection process")

        input_path = Path(input_folder)
        output_path = Path(output_base) / input_path.name
        output_path.mkdir(parents=True, exist_ok=True)

        # Supported image extensions
        exts = [".jpg", ".jpeg", ".png", ".bmp"]

        for img_file in input_path.glob("*"):
            if img_file.suffix.lower() not in exts:
                continue

            try:
                img = cv2.imread(str(img_file))
                if img is None:
                    print(f"Could not read {img_file}")
                    continue

                # Detect faces
                faces = RetinaFace.detect_faces(str(img_file))
                if not isinstance(faces, dict):
                    print(f" No faces found in {img_file}")
                    continue

                count = 0
                for key, face in faces.items():
                    x1, y1, x2, y2 = map(int, face["facial_area"])
                    face_crop = img[y1:y2, x1:x2]

                    save_name = f"{img_file.stem}_face{count+1}.jpg"
                    save_path = output_path / save_name
                    cv2.imwrite(str(save_path), face_crop)

                    count += 1
                    if max_faces and count >= max_faces:
                        break

                print(f"{img_file.name}: {count} faces saved to {output_path}")

            except Exception as e:
                print(f"Error processing {img_file}: {e}")

        logging.info("Finished selecting images")
        return output_path

    def main(self,path):
        
        input_folder = path   # folder containing images
        output_base = r"C:\ht\outputs"                           
        max_faces = None                                       

        path_to_return=self.extract_faces_from_folder(input_folder, output_base, max_faces)
        return path_to_return


if __name__ == "__main__":
    ip=r"C:\ht\raw_frames\2025_09_17_00_54_31"
    obj = ImageSelector()
    obj.main(path=ip)
