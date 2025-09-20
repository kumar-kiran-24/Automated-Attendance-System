import cv2
import datetime


# Example: rtsp://username:password@ip_address:port/stream
video_source = 0   # 0 = webcam, replace with CCTV RTSP or file
cap = cv2.VideoCapture(video_source)

# Video writer (only initialized when recording starts)
out = None
recording = False

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("CCTV Feed", frame)

    key = cv2.waitKey(1) & 0xFF

    # Start recording when 'r' is pressed
    if key == ord('r') and not recording:
        recording = True
        # Unique filename with timestamp
        filename = datetime.datetime.now().strftime("clip_%Y%m%d_%H%M%S.avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(filename, fourcc, 20.0,
                              (frame.shape[1], frame.shape[0]))
        print(f"Recording started: {filename}")

    # Stop recording when 's' is pressed
    elif key == ord('s') and recording:
        recording = False
        out.release()
        out = None
        print("Recording stopped and saved.")

    # Write frames to file if recording
    if recording and out is not None:
        out.write(frame)

    # Quit with 'q'
    if key == ord('q'):
        break

cap.release()
if out is not None:
    out.release()
cv2.destroyAllWindows()
