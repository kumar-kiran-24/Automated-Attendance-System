import cv2
import os
from datetime import datetime, timedelta

def record_video(save_path="./videos", duration=300, fps=20, resolution=(640, 480)):
    # Ensure save directory exists
    os.makedirs(save_path, exist_ok=True)

    # Get start and end time
    start_time = datetime.now()
    end_time = start_time + timedelta(seconds=duration)

    # Format filename: DD_MM_YY__HH-MM__HH-MM.avi  (using AVI for better compatibility)
    filename = f"{start_time.strftime('%d_%m_%y')}__{start_time.strftime('%H-%M')}__{end_time.strftime('%H-%M')}.avi"
    full_path = os.path.join(save_path, filename)

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Could not open camera.")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

    # Define codec (XVID for AVI)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(full_path, fourcc, fps, resolution)

    if not out.isOpened():
        print("❌ Error: Could not open video writer.")
        return

    print(f"🎥 Recording started... File will be saved as {full_path}")

    while (datetime.now() - start_time).total_seconds() < duration:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Warning: Failed to read frame from camera.")
            break
        out.write(frame)

        cv2.imshow('Recording', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Recording stopped manually.")
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"✅ Recording finished. Video saved as {full_path}")


# Example: save in a custom folder
record_video(save_path="C:/ht/video_record/database", duration=10)
