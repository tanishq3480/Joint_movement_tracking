import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ---- LOAD MODEL ----
model_path = r"C:\Python\models\pose_landmarker_full.task"

options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=model_path),
    running_mode=vision.RunningMode.VIDEO,
    num_poses=1
)

detector = vision.PoseLandmarker.create_from_options(options)

# ---- ANGLE FUNCTION ----
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360-angle if angle > 180 else angle

# ---- MOVEMENT CLASSIFICATION ----
def classify_movement(angle, joint):
    if joint in ["elbow", "knee"]:
        if angle < 60:
            return "Flexion (Sagittal Plane)"
        elif angle > 150:
            return "Extension (Sagittal Plane)"
    if joint == "shoulder":
        if angle < 70:
            return "Adduction"
        elif angle > 120:
            return "Abduction"
    if joint == "hip":
        if angle < 70:
            return "Flexion"
        elif angle > 150:
            return "Extension"
    return "Intermediate"

# ---- ROM STORAGE ----
rom = {
    "elbow": [180, 0],
    "knee": [180, 0],
    "shoulder": [180, 0],
    "hip": [180, 0]
}

# ---- CAMERA ----
cap = cv2.VideoCapture(0)
frame_id = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    result = detector.detect_for_video(mp_img, frame_id)

    if result.pose_landmarks:
        lm = result.pose_landmarks[0]

        def pt(i):
            return [int(lm[i].x * w), int(lm[i].y * h)]

        # ---- JOINTS ----
        shoulder = pt(11)
        elbow = pt(13)
        wrist = pt(15)

        hip = pt(23)
        knee = pt(25)
        ankle = pt(27)

        shoulder_r = pt(12)
        hip_r = pt(24)

        # ---- DRAW POINTS ----
        for p in [shoulder, elbow, wrist, hip, knee, ankle]:
            cv2.circle(frame, tuple(p), 6, (0,255,0), -1)

        # ---- ANGLES ----
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        knee_angle = calculate_angle(hip, knee, ankle)
        shoulder_angle = calculate_angle(elbow, shoulder, hip)
        hip_angle = calculate_angle(shoulder, hip, knee)

        # ---- ROM UPDATE ----
        for name, val in [("elbow", elbow_angle), ("knee", knee_angle),
                          ("shoulder", shoulder_angle), ("hip", hip_angle)]:
            rom[name][0] = min(rom[name][0], val)
            rom[name][1] = max(rom[name][1], val)

        # ---- MOVEMENT TYPES ----
        elbow_type = classify_movement(elbow_angle, "elbow")
        knee_type = classify_movement(knee_angle, "knee")
        shoulder_type = classify_movement(shoulder_angle, "shoulder")
        hip_type = classify_movement(hip_angle, "hip")

        # ---- PELVIS ROTATION ----
        pelvis_dx = hip_r[0] - hip[0]
        pelvis_dy = hip_r[1] - hip[1]
        pelvis_angle = np.degrees(np.arctan2(pelvis_dy, pelvis_dx))

        # ---- DISPLAY ----
        y = 30
        for text in [
            f'Elbow: {int(elbow_angle)} | {elbow_type}',
            f'Knee: {int(knee_angle)} | {knee_type}',
            f'Shoulder: {int(shoulder_angle)} | {shoulder_type}',
            f'Hip: {int(hip_angle)} | {hip_type}',
            f'Pelvis Rotation: {int(pelvis_angle)} deg',
            f'Elbow ROM: {int(rom["elbow"][0])}-{int(rom["elbow"][1])}',
            f'Knee ROM: {int(rom["knee"][0])}-{int(rom["knee"][1])}'
        ]:
            cv2.putText(frame, text, (20, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 2)
            y += 25

    cv2.imshow("Full Biomechanics Analyzer", frame)
    frame_id += 1

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()