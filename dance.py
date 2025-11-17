import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def get_point(landmarks, index, w, h):
    lm = landmarks[index]
    return int(lm.x * w), int(lm.y * h)

def draw_avatar(canvas, k):
    head = k["head"]
    el = k["elbow_l"]; er = k["elbow_r"]
    wl = k["wrist_l"]; wr = k["wrist_r"]
    kl = k["knee_l"]; kr = k["knee_r"]
    al = k["ankle_l"]; ar = k["ankle_r"]

    # ---------------------------------------------------
    # FIXED torso size (constant body)
    # ---------------------------------------------------
    torso_width = 120
    torso_height = 180

    # Torso top starts just below the head
    torso_top = (head[0], head[1] + 40)
    torso_left = (torso_top[0] - torso_width // 2, torso_top[1])
    torso_right = (torso_top[0] + torso_width // 2, torso_top[1])
    torso_bottom = (torso_top[0], torso_top[1] + torso_height)

    # ---------------------------------------------------
    # Neck (short connector)
    # ---------------------------------------------------
    cv2.line(canvas, (head[0], head[1] + 35), torso_top, (0, 0, 0), 5)

    # ---------------------------------------------------
    # Head
    # ---------------------------------------------------
    cv2.circle(canvas, head, 35, (255, 220, 180), -1)

    # Eyes
    cv2.circle(canvas, (head[0] - 10, head[1] - 10), 5, (0, 0, 0), -1)
    cv2.circle(canvas, (head[0] + 10, head[1] - 10), 5, (0, 0, 0), -1)

    # Mouth
    cv2.line(canvas, (head[0] - 10, head[1] + 15), (head[0] + 10, head[1] + 15), (0, 0, 0), 3)

    # Hair
    cv2.line(canvas, (head[0] - 10, head[1] - 30), (head[0] - 5, head[1] - 50), (0, 0, 0), 4)
    cv2.line(canvas, (head[0] + 10, head[1] - 30), (head[0] + 5, head[1] - 50), (0, 0, 0), 4)

    # ---------------------------------------------------
    # Torso (constant rectangle)
    # ---------------------------------------------------
    cv2.rectangle(canvas, torso_left, (torso_right[0], torso_bottom[1]), (200, 200, 255), -1)
    cv2.rectangle(canvas, torso_left, (torso_right[0], torso_bottom[1]), (0, 0, 0), 4)

    # Torso left/right anchor points for arms
    torso_left_arm = (torso_left[0], torso_top[1] + 40)
    torso_right_arm = (torso_right[0], torso_top[1] + 40)

    # Torso left/right leg joints
    torso_left_leg = (torso_left[0] + 30, torso_bottom[1])
    torso_right_leg = (torso_right[0] - 30, torso_bottom[1])

    # ---------------------------------------------------
    # Arms (connect to torso, NOT shoulders)
    # ---------------------------------------------------
    cv2.line(canvas, torso_left_arm, el, (0, 0, 0), 5)
    cv2.line(canvas, el, wl, (0, 0, 0), 5)

    cv2.line(canvas, torso_right_arm, er, (0, 0, 0), 5)
    cv2.line(canvas, er, wr, (0, 0, 0), 5)

    # Hands
    cv2.circle(canvas, wl, 10, (0, 0, 0), -1)
    cv2.circle(canvas, wr, 10, (0, 0, 0), -1)

    # ---------------------------------------------------
    # Legs (connect to torso bottom)
    # ---------------------------------------------------
    cv2.line(canvas, torso_left_leg, kl, (0, 0, 0), 6)
    cv2.line(canvas, kl, al, (0, 0, 0), 6)

    cv2.line(canvas, torso_right_leg, kr, (0, 0, 0), 6)
    cv2.line(canvas, kr, ar, (0, 0, 0), 6)

    # Feet
    cv2.circle(canvas, al, 12, (0, 0, 0), -1)
    cv2.circle(canvas, ar, 12, (0, 0, 0), -1)


# ---------------------------------------------------
# Live tracking + avatar windows
# ---------------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb)

    # White background for avatar
    avatar = np.ones((720, 720, 3), dtype=np.uint8) * 255

    if result.pose_landmarks:
        lm = result.pose_landmarks.landmark

        k = {
            "head": get_point(lm, mp_pose.PoseLandmark.NOSE, w, h),
            "elbow_l": get_point(lm, mp_pose.PoseLandmark.LEFT_ELBOW, w, h),
            "elbow_r": get_point(lm, mp_pose.PoseLandmark.RIGHT_ELBOW, w, h),
            "wrist_l": get_point(lm, mp_pose.PoseLandmark.LEFT_WRIST, w, h),
            "wrist_r": get_point(lm, mp_pose.PoseLandmark.RIGHT_WRIST, w, h),
            "knee_l": get_point(lm, mp_pose.PoseLandmark.LEFT_KNEE, w, h),
            "knee_r": get_point(lm, mp_pose.PoseLandmark.RIGHT_KNEE, w, h),
            "ankle_l": get_point(lm, mp_pose.PoseLandmark.LEFT_ANKLE, w, h),
            "ankle_r": get_point(lm, mp_pose.PoseLandmark.RIGHT_ANKLE, w, h),
        }

        # Draw avatar
        draw_avatar(avatar, k)

        # Tracking window: draw skeleton instead of dots
        connections = [
            (11, 13), (13, 15),  # Left arm
            (12, 14), (14, 16),  # Right arm
            (23, 25), (25, 27),  # Left leg
            (24, 26), (26, 28),  # Right leg
            (11, 12), (23, 24),  # Shoulders + hips
            (11, 23), (12, 24)   # Sides
        ]

        for a, b in connections:
            x1 = int(lm[a].x * w); y1 = int(lm[a].y * h)
            x2 = int(lm[b].x * w); y2 = int(lm[b].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

    cv2.imshow("Tracking View", frame)
    cv2.imshow("Dancing Avatar", avatar)

    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
