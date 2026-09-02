import time
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# ==============================================================================
# CONFIGURATION
# ==============================================================================
MODEL_PATH = "runs/detect/chess_piece_detector-5/weights/best.pt"
BOARD_PHYSICAL_SIZE_MM = 293.0
BLOCK_W_X = BOARD_PHYSICAL_SIZE_MM / 8.0
BLOCK_W_Y = BOARD_PHYSICAL_SIZE_MM / 8.0

model = YOLO(MODEL_PATH)

# Dual-stream setup matching your hardware configuration
mainstream = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)  # 15 or 30 fps
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
profile = mainstream.start(config)
color_sensor = profile.get_device().first_color_sensor()
color_sensor.set_option(rs.option.enable_auto_exposure, 1)
color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
# color_sensor.set_option(rs.option.white_balance, 3400.0)
# color_sensor.set_option(rs.option.exposure, 250.0)

# # Lower gain slightly to reduce sensor noise
# color_sensor.set_option(rs.option.gain, 50.0)
alinger = rs.align(rs.stream.color)
colorizer = rs.colorizer()

alinger = rs.align(rs.stream.color)

print("Warming up camera sensor...")
for _ in range(20):
    mainstream.wait_for_frames()

# ==============================================================================
# HOMOGRAPHY CALIBRATION (Retry loop)
# ==============================================================================
coner_coordinates = []
print("Searching for 4 red corner markers...")

while len(coner_coordinates) != 4:
    frames = mainstream.wait_for_frames(50000)
    aligned_frames = alinger.process(frames)
    actual_color_data = np.asanyarray(aligned_frames.get_color_frame().get_data())

    hsv = cv2.cvtColor(actual_color_data, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 120, 100])
    upper_red1 = np.array([6, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

    lower_red2 = np.array([172, 120, 100])
    upper_red2 = np.array([179, 255, 255])
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    coner_coordinates = []
    for c in contours:
        if cv2.contourArea(c) > 15:
            M = cv2.moments(c)
            if M["m00"] != 0:
                coner_coordinates.append([M["m10"] / M["m00"], M["m01"] / M["m00"]])

def order_corner_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts_arr = np.array(pts, dtype="float32")
    
    s = pts_arr.sum(axis=1)
    rect[0] = pts_arr[np.argmin(s)]  # Top-left (smallest x + y)
    rect[2] = pts_arr[np.argmax(s)]  # Bottom-right (largest x + y)

    diff = np.diff(pts_arr, axis=1)
    rect[1] = pts_arr[np.argmin(diff)]  # Top-right (smallest x - y)
    rect[3] = pts_arr[np.argmax(diff)]  # Bottom-left (largest x - y)

    return rect

src_pts = order_corner_points(coner_coordinates)
dst_pts = np.float32([[0.0, 0.0], [293.0, 0.0], [293.0, 293.0], [0.0, 293.0]])
H_flatten = cv2.getPerspectiveTransform(src_pts, dst_pts)

# coner_coordinates = sorted(coner_coordinates, key=lambda p: p[1])
# top_pts = sorted(coner_coordinates[:2], key=lambda p: p[0])
# bottom_pts = sorted(coner_coordinates[2:], key=lambda p: p[0])

# src_pts = np.float32([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]])
# dst_pts = np.float32([[0.0, 0.0], [293.0, 0.0], [293.0, 293.0], [0.0, 293.0]])

H_flatten = cv2.getPerspectiveTransform(src_pts, dst_pts)


def flatten_point(pixel_x, pixel_y):
    """Converts camera pixels to board millimeters."""
    pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    flattened = cv2.perspectiveTransform(pt, H_flatten)
    return flattened[0][0][0], flattened[0][0][1]


def get_board_occupancy():
    """Runs YOLO and returns the 8x8 binary occupancy matrix + annotated video frame."""
    frames = mainstream.wait_for_frames(50000)
    aligned = alinger.process(frames)
    frame = np.asanyarray(aligned.get_color_frame().get_data())

    matrix = np.zeros((8, 8), dtype=int)
    display_frame = frame.copy()

    results = model(frame, conf=0.05, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        mm_x, mm_y = flatten_point(cx, cy)

        col = int(mm_x // BLOCK_W_X)
        row = int(mm_y // BLOCK_W_Y)

        if 0 <= col < 8 and 0 <= row < 8:
            matrix[row][col] = 1

            # Visual overlay: Draw bounding box center and grid coordinates
            cv2.circle(display_frame, (int(cx), int(cy)), 5, (0, 255, 0), -1)
            cv2.putText(
                display_frame,
                f"[{row},{col}]",
                (int(cx) + 5, int(cy) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
            )

    return matrix, display_frame


# ==============================================================================
# TEST LOOP
# ==============================================================================
print("\n✅ Setup complete! Press 's' to print board matrix | Press 'q' to quit.")

try:
    while True:
        matrix, view = get_board_occupancy()
        cv2.imshow("Vision Occupancy Test", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            print("\nCaptured 8x8 Board Matrix (1 = Piece Present, 0 = Empty):")
            print(matrix)
            print("-" * 45)
        elif key == ord("q"):
            break
finally:
    mainstream.stop()
    cv2.destroyAllWindows()