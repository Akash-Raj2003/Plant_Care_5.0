import cv2
import numpy as np
import time
import threading
from flask import Flask, Response, jsonify
from picamera2 import Picamera2


# CONFIG

ARUCO_TYPE = "DICT_7X7_50"
MARKER_SIZE = 0.036  # meters
CALIBRATION_FILE = "calibration_chessboard_pi.yaml"

FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 15

JPEG_QUALITY = 60   # debug stream

ARUCO_DICT = {
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50
}


# LOAD CAMERA CALIBRATION

cv_file = cv2.FileStorage(CALIBRATION_FILE, cv2.FILE_STORAGE_READ)
camera_matrix = cv_file.getNode("K").mat()
dist_coeffs = cv_file.getNode("D").mat()
cv_file.release()

if camera_matrix is None or dist_coeffs is None:
    raise RuntimeError(f"Failed to load calibration from {CALIBRATION_FILE}")

print(" Camera calibration loaded")

# ARUCO SETUP

if ARUCO_TYPE not in ARUCO_DICT:
    raise RuntimeError(f"Unsupported ArUco type: {ARUCO_TYPE}")

aruco_dict = cv2.aruco.getPredefinedDictionary(ARUCO_DICT[ARUCO_TYPE])
aruco_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

obj_points = np.array([
    [-MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2,  MARKER_SIZE / 2, 0],
    [ MARKER_SIZE / 2, -MARKER_SIZE / 2, 0],
    [-MARKER_SIZE / 2, -MARKER_SIZE / 2, 0]
], dtype=np.float32)

# SHARED STATE

latest_pose = {
    "timestamp": 0.0,
    "marker_found": False,
    "marker_id": None,
    "x": None,
    "y": None,
    "z": None,
    "distance": None,
    "rvec": None,
    "tvec": None
}

latest_debug_frame = None
state_lock = threading.Lock()

# camera thread will update latest_pose and latest_debug_frame with state_lock protection

def choose_marker(marker_poses):
    if len(marker_poses) == 0:
        return None
    distances = [np.linalg.norm(tvec.flatten()) for (_, _, _, tvec) in marker_poses]
    return int(np.argmin(distances))

def camera_loop():
    global latest_pose, latest_debug_frame

    picam2 = Picamera2()
    config = picam2.create_video_configuration(
        main={"size": (FRAME_WIDTH, FRAME_HEIGHT), "format": "RGB888"},
        controls={"FrameRate": FRAME_RATE}
    )
    picam2.configure(config)
    picam2.start()

    print("[INFO] Pi camera started")

    while True:
        frame = picam2.capture_array()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        output = frame_bgr.copy()
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = detector.detectMarkers(gray)

        pose_data = {
            "timestamp": time.time(),
            "marker_found": False,
            "marker_id": None,
            "x": None,
            "y": None,
            "z": None,
            "distance": None,
            "rvec": None,
            "tvec": None
        }

        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(output, corners)
            marker_poses = []

            for i in range(len(ids)):
                img_points = corners[i][0].astype(np.float32)

                success, rvec, tvec = cv2.solvePnP(
                    obj_points,
                    img_points,
                    camera_matrix,
                    dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )

                if success:
                    marker_poses.append((i, int(ids[i][0]), rvec, tvec))

            if len(marker_poses) > 0:
                selected_idx = choose_marker(marker_poses)

                for j, (corner_idx, marker_id, rvec, tvec) in enumerate(marker_poses):
                    distance = float(np.linalg.norm(tvec))
                    x, y, z = tvec.flatten()

                    cv2.drawFrameAxes(
                        output,
                        camera_matrix,
                        dist_coeffs,
                        rvec,
                        tvec,
                        MARKER_SIZE * 0.5
                    )

                    pts = corners[corner_idx][0].astype(int)
                    top_left = tuple(pts[0])

                    color = (0, 255, 0) if j == selected_idx else (180, 180, 0)

                    cv2.putText(
                        output,
                        f"ID:{marker_id} Dist:{distance:.3f}m",
                        (top_left[0], max(20, top_left[1] - 30)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        color,
                        2
                    )

                    cv2.putText(
                        output,
                        f"X:{x:.3f} Y:{y:.3f} Z:{z:.3f}",
                        (top_left[0], max(40, top_left[1] - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.50,
                        color,
                        2
                    )

                _, marker_id, rvec, tvec = marker_poses[selected_idx]
                x, y, z = tvec.flatten()
                distance = float(np.linalg.norm(tvec))

                pose_data = {
                    "timestamp": time.time(),
                    "marker_found": True,
                    "marker_id": marker_id,
                    "x": float(x),
                    "y": float(y),
                    "z": float(z),
                    "distance": distance,
                    "rvec": [float(v) for v in rvec.flatten()],
                    "tvec": [float(v) for v in tvec.flatten()]
                }

        with state_lock:
            latest_pose = pose_data
            latest_debug_frame = output

# flask app to serve latest pose as JSON and debug video feed as MJPEG

app = Flask(__name__)

@app.route("/pose", methods=["GET"])
def get_pose():
    with state_lock:
        return jsonify(latest_pose)

def mjpeg_generator():
    global latest_debug_frame
    while True:
        with state_lock:
            frame = None if latest_debug_frame is None else latest_debug_frame.copy()

        if frame is None:
            time.sleep(0.02)
            continue

        ok, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        )
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            jpeg.tobytes() +
            b"\r\n"
        )
        time.sleep(0.03)

@app.route("/video_feed")
def video_feed():
    return Response(
        mjpeg_generator(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )

# MAIN

if __name__ == "__main__":
    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    print("[INFO] Flask server starting on port 5000")
    app.run(host="0.0.0.0", port=5000, threaded=True)