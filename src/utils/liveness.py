import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis

# Initialize FaceAnalysis
app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# ---------- Blink Detection (using EAR) ----------
def eye_aspect_ratio(eye_points):
    # eye_points is 6 landmarks (x,y)
    A = np.linalg.norm(eye_points[1] - eye_points[5])
    B = np.linalg.norm(eye_points[2] - eye_points[4])
    C = np.linalg.norm(eye_points[0] - eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

EYE_AR_THRESH = 0.2

def detect_blink(landmarks):
    # InsightFace gives 106 landmarks, we pick eyes
    left_eye_idx = [60, 64, 62, 66, 68, 70]   # approx eye outline points
    right_eye_idx = [96, 100, 98, 102, 104, 106]

    left_eye = np.array([landmarks[i] for i in left_eye_idx])
    right_eye = np.array([landmarks[i] for i in right_eye_idx])

    leftEAR = eye_aspect_ratio(left_eye)
    rightEAR = eye_aspect_ratio(right_eye)
    ear = (leftEAR + rightEAR) / 2.0

    return ear < EYE_AR_THRESH

# ---------- Head Pose Estimation ----------
def detect_head_movement(landmarks):
    # Use nose tip (landmark 52) relative to center
    nose = landmarks[52]
    if abs(nose[0] - 0.5) > 0.05 or abs(nose[1] - 0.5) > 0.05:
        return True
    return False

# ---------- Main loop ----------
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = app.get(frame)
    if faces:
        for face in faces:
            # Draw bounding box
            x1, y1, x2, y2 = face.bbox.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Extract landmarks (106)
            lmk = face.landmark_2d_106

            blinked = detect_blink(lmk)
            moved = detect_head_movement(lmk)

            if blinked or moved:
                cv2.putText(frame, "REAL", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            else:
                cv2.putText(frame, "SPOOF?", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow("Liveness Detection - InsightFace", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
