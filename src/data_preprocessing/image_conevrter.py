import os
import re
import cv2
from retinaface import RetinaFace


def get_safe_filename(name: str) -> str:
    """
    Sanitize filenames/folders for Windows.
    Replace invalid characters with "_".
    """
    return re.sub(r'[^A-Za-z0-9._-]', '_', name)


def create_output_folders(video_path: str):
    """
    Create main output folder for the video, with two subfolders:
    1. raw_frames → all frames
    2. selected_frames → frames with most faces
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_name = get_safe_filename(video_name)

    base_output = os.path.join("outputs", video_name)
    raw_dir = os.path.join("raw_frames", "raw_frames")
    selected_dir = os.path.join(base_output, "selected_frames")

    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(selected_dir, exist_ok=True)

    return raw_dir, selected_dir


def detect_faces(frame):
    """
    Detect faces using RetinaFace.
    Returns number of faces detected.
    """
    detections = RetinaFace.detect_faces(frame)
    return len(detections) if isinstance(detections, dict) else 0


def save_frame(frame, path):
    """
    Save a single frame to disk.
    """
    cv2.imwrite(path, frame)


def process_video(video_path, min_faces=0):
    """
    Main function:
    - Save all raw frames
    - Save frames with >= min_faces and >= max faces seen so far
    """
    raw_dir, selected_dir = create_output_folders(video_path)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_selected = 0
    max_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Save raw frame
        raw_path = os.path.join(raw_dir, f"frame_{frame_count}.jpg")
        save_frame(frame, raw_path)

        # Detect faces
        num_faces = detect_faces(frame)

        # Save "selected" frame if it meets criteria
        if num_faces >= min_faces and num_faces >= max_faces:
            max_faces = num_faces
            sel_path = os.path.join(selected_dir, f"frame_{frame_count}_faces_{num_faces}.jpg")
            save_frame(frame, sel_path)
            saved_selected += 1
            print(f"[SAVED] Frame {frame_count} with {num_faces} faces → {sel_path}")

    cap.release()
    print(f"\n✅ Process completed!\n"
          f"📂 Raw frames: {raw_dir}\n"
          f"📂 Selected frames: {selected_dir}\n"
          f"Total selected: {saved_selected}")


if __name__ == "__main__":
    video_path = r"captured_data\video\2025_09_16_13_49_20.avi"  # raw string to avoid escape bugs
    process_video(video_path, min_faces=0)
