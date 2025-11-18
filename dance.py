import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

alpha = 0.3   # smoothing
prev = {}

def smooth(name, new):
    if name not in prev:
        prev[name] = new
        return new
    
    px, py = prev[name]
    nx, ny = new
    sx = int(px + alpha * (nx - px))
    sy = int(py + alpha * (ny - py))
    prev[name] = (sx, sy)
    return (sx, sy)


# ----------------------------------------------------------
# FIXED LANDMARK EXTRACTION WITH LEFT/RIGHT CORRECTION
# ----------------------------------------------------------
def get_landmarks(frame):
    h, w, _ = frame.shape
    res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if not res.pose_landmarks:
        return None

    lm = res.pose_landmarks.landmark

    def P(id):
        return smooth(str(id), (int(lm[id].x * w), int(lm[id].y * h)))

    pts = {
        "H": P(mp_pose.PoseLandmark.NOSE),
        "LS": P(mp_pose.PoseLandmark.LEFT_SHOULDER),
        "RS": P(mp_pose.PoseLandmark.RIGHT_SHOULDER),
        "LE": P(mp_pose.PoseLandmark.LEFT_ELBOW),
        "RE": P(mp_pose.PoseLandmark.RIGHT_ELBOW),
        "LW": P(mp_pose.PoseLandmark.LEFT_WRIST),
        "RW": P(mp_pose.PoseLandmark.RIGHT_WRIST),
        "LH": P(mp_pose.PoseLandmark.LEFT_HIP),
        "RH": P(mp_pose.PoseLandmark.RIGHT_HIP),
        "LK": P(mp_pose.PoseLandmark.LEFT_KNEE),
        "RK": P(mp_pose.PoseLandmark.RIGHT_KNEE),
        "LA": P(mp_pose.PoseLandmark.LEFT_ANKLE),
        "RA": P(mp_pose.PoseLandmark.RIGHT_ANKLE)
    }

    # Fix shoulder left/right
    if pts["LS"][0] > pts["RS"][0]:
        pts["LS"], pts["RS"] = pts["RS"], pts["LS"]

    # Wrist sanity fix
    if pts["LW"][0] > pts["RS"][0] + 40:
        pts["LW"], pts["RW"] = pts["RW"], pts["LW"]
        pts["LE"], pts["RE"] = pts["RE"], pts["LE"]

    if pts["RW"][0] < pts["LS"][0] - 40:
        pts["LW"], pts["RW"] = pts["RW"], pts["LW"]
        pts["LE"], pts["RE"] = pts["RE"], pts["LE"]

    return pts


# ----------------------------------------------------------
# CARTOON CHARACTER DRAWING (FIXED ATTACHMENT)
# ----------------------------------------------------------
def draw_simple_character(img, p):

    # ---------- HEAD ----------
    head_center = p["H"]
    head_radius = 40
    cv2.circle(img, head_center, head_radius, (255, 225, 200), -1)
    cv2.circle(img, head_center, head_radius, (0,0,0), 2)

    # Eyes
    cv2.circle(img, (head_center[0]-12, head_center[1]-8), 6, (0,0,0), -1)
    cv2.circle(img, (head_center[0]+12, head_center[1]-8), 6, (0,0,0), -1)

    # Smile
    cv2.ellipse(img, (head_center[0], head_center[1]+12), (15, 7), 0, 0, 180, (0,0,0), 2)

    # Simple hair
    cv2.line(img, (head_center[0]-18, head_center[1]-38), (head_center[0]-5, head_center[1]-48), (0,0,0), 3)
    cv2.line(img, (head_center[0]+18, head_center[1]-38), (head_center[0]+5, head_center[1]-48), (0,0,0), 3)

    # ---------- NECK ----------
    mid_shoulder = ((p["LS"][0] + p["RS"][0])//2, (p["LS"][1] + p["RS"][1])//2)
    neck_top = (head_center[0], head_center[1] + head_radius)
    neck_bottom = (mid_shoulder[0], mid_shoulder[1])
    cv2.line(img, neck_top, neck_bottom, (255, 225, 200), 6)

    # ---------- BODY ----------
    mid_hip = ((p["LH"][0] + p["RH"][0])//2, (p["LH"][1] + p["RH"][1])//2)

    torso_left = mid_shoulder[0] - 30
    torso_right = mid_shoulder[0] + 30
    torso_top = mid_shoulder[1]
    torso_bottom = mid_hip[1]

    cv2.rectangle(img, (torso_left, torso_top),
                  (torso_right, torso_bottom),
                  (0, 150, 255), -1)

    # ---------- ARMS ----------
    cv2.line(img, p["LS"], p["LE"], (255,225,200), 10)
    cv2.line(img, p["LE"], p["LW"], (255,225,200), 8)
    cv2.circle(img, p["LW"], 8, (0,0,0), -1)

    cv2.line(img, p["RS"], p["RE"], (255,225,200), 10)
    cv2.line(img, p["RE"], p["RW"], (255,225,200), 8)
    cv2.circle(img, p["RW"], 8, (0,0,0), -1)

    # ---------- LEGS ----------
    cv2.line(img, p["LH"], p["LK"], (255,225,200), 12)
    cv2.line(img, p["LK"], p["LA"], (255,225,200), 10)
    cv2.circle(img, p["LA"], 8, (0,0,0), -1)

    cv2.line(img, p["RH"], p["RK"], (255,225,200), 12)
    cv2.line(img, p["RK"], p["RA"], (255,225,200), 10)
    cv2.circle(img, p["RA"], 8, (0,0,0), -1)


# ----------------------------------------------------------
# MAIN LOOP
# ----------------------------------------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    pts = get_landmarks(frame)

    # -----------------------------------
    # TRACKING WINDOW
    # -----------------------------------
    tracking = frame.copy()
    if pts:
        # Draw datapoints
        for k, v in pts.items():
            cv2.circle(tracking, v, 6, (0,255,0), -1)

        # Lines between joints
        lines = [
            ("LS","RS"),
            ("LS","LE"),("LE","LW"),
            ("RS","RE"),("RE","RW"),
            ("LH","RH"),
            ("LH","LK"),("LK","LA"),
            ("RH","RK"),("RK","RA")
        ]

        for a,b in lines:
            cv2.line(tracking, pts[a], pts[b], (0,255,0), 2)

    cv2.imshow("Tracking", tracking)

    # -----------------------------------
    # CHARACTER WINDOW
    # -----------------------------------
    canvas = np.zeros_like(frame) + 255
    if pts:
        draw_simple_character(canvas, pts)
    cv2.imshow("Character", canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
