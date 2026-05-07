import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose()
hands = mp_hands.Hands(max_num_hands=2)

cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    angle = np.degrees(
        np.arccos(
            np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        )
    )
    return angle

# ROM trackers
elbow_min, elbow_max = 180, 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results_pose = pose.process(image)
    results_hands = hands.process(image)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results_pose.pose_landmarks:
        landmarks = results_pose.pose_landmarks.landmark

        # Get coordinates
        shoulder = [landmarks[11].x, landmarks[11].y]
        elbow = [landmarks[13].x, landmarks[13].y]
        wrist = [landmarks[15].x, landmarks[15].y]

        # Calculate elbow angle
        angle = calculate_angle(shoulder, elbow, wrist)

        # Update ROM
        elbow_min = min(elbow_min, angle)
        elbow_max = max(elbow_max, angle)

        # Movement classification
        movement = "Static"
        if angle < 160:
            movement = "Flexion"
        elif angle > 160:
            movement = "Extension"

        # Display
        cv2.putText(image, f'Angle: {int(angle)}',
                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,255,0), 2)

        cv2.putText(image, f'Movement: {movement}',
                    (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255,0,0), 2)

        cv2.putText(image, f'ROM: {int(elbow_max - elbow_min)}',
                    (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2)

        mp_draw.draw_landmarks(image, results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Hand tracking (for fingers)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ROM Analyzer", image)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()