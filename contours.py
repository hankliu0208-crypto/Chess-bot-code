import time
import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import threading
import chess
import chess.engine
from ultralytics import YOLO








XARM_IP = "192.168.1.222"

HOME_X = 290.0  
HOME_Y = 10.0 
HOME_Z = 550.0  
HOME_ROLL = 180
HOME_PITCH = -0.4
HOME_YAW = -1.28
SCREEN_CX, SCREEN_CY =320, 240
PIXEL_TOLERANCE = 3  






BOARD_PHYSICAL_SIZE_MM = 293.0

# --- 2. BOARD BOUNDARIES IN PIXELS ---

LAST_MOVE_TIME = 0.0
MOVE_INTERVAL = 0.20 
CHESS_BOARD_WIDTH = []
CHESS_BOARD_HEIGHT =[]
TOLERACE = 50

SPEED = 60
set_up = True


pieces = {
    "rook": 170,
    "queen": 190,
    "bishop": 175,
    "pawn": 160,
    "knight": 162,
    "king": 190
}

for i in pieces.keys():
    pieces[i] += 40
found = True






# OPERATION = [
#     [[1, 1], [1, 8]],  # Rook 1
#     [[2, 1], [2, 8]],  # Knight 1
#     [[3, 1], [3, 8]],  # Bishop 1
#     [[4, 1], [4, 8]],  # Queen
#     [[5, 1], [5, 8]],  # King
#     [[6, 1], [6, 8]],  # Bishop 2
#     [[7, 1], [7, 8]],  # Knight 2
#     [[8, 1], [8, 8]],  # Rook 2
# ]

piece = [
    "rook",
    "knight",
    "bishop",
    "queen",
    "king",
    "bishop",
    "knight",
    "rook",
]





OPERATION = []  

MODEL_PATH = "runs/detect/chess_piece_detector-5/weights/best.pt"
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"

engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)

model = YOLO(MODEL_PATH)
board_logic = chess.Board()



piece = []
coner_coordinates = []



origninal = OPERATION
print("Connecting...")
arm = XArmAPI(XARM_IP, is_check_robot_settings=False) 

time.sleep(1.0) 
def open_claw():
    print("--> Opening Claw...")
    print(arm.set_cgpio_digital(ionum=8, value=0))  # DO0 ON
    time.sleep(0.2)
    print(arm.set_cgpio_digital(ionum=9, value=1))  # DO1 OFF
    time.sleep(1.0)

def close_claw():
    print("--> Closing Claw...")
    print(arm.set_cgpio_digital(ionum=8, value=1))  
    time.sleep(0.2)
    print(arm.set_cgpio_digital(ionum=9, value=0))  
    time.sleep(1.0)

arm.clean_error()
arm.clean_warn()
time.sleep(0.2)

# Enable motors
arm.motion_enable(enable=True)
arm.set_mode(0)
arm.set_state(state=0)

print("Waiting for xArm to enter State 0...")


open_claw()
time.sleep(2.0) 
print("State check:", arm.get_state()) 


arm.set_position(HOME_X, HOME_Y, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
print("State check:", arm.get_state()) 

mainstream = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
profile = mainstream.start(config)

color_sensor = profile.get_device().first_color_sensor()

# Turn OFF auto-exposure
color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.enable_auto_white_balance, 1)


# Set manual exposure (Range is typically 1 to 10000; ~50-150 prevents washout)

color_sensor.set_option(rs.option.exposure, 250.0)
# color_sensor.set_option(rs.option.white_balance, 3400.0)

# # Lower gain slightly to reduce sensor noise
# color_sensor.set_option(rs.option.gain, 50.0)
alinger = rs.align(rs.stream.color)
colorizer = rs.colorizer()
for _ in range(30):
        mainstream.wait_for_frames()
print("searching for red dots...")
while len(coner_coordinates) != 4:

    

    frames = mainstream.wait_for_frames(50000)
    aligned_frames = alinger.process(frames)
    actual_color_data = np.asanyarray(aligned_frames.get_color_frame().get_data())
    actual_depth_data = np.asanyarray(aligned_frames.get_depth_frame().get_data())





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



    contours, _ = cv2.findContours(
        red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )






    output_view = actual_color_data.copy()
    for contour in contours:
        if cv2.contourArea(contour) > 50:  
            cv2.drawContours(output_view, [contour], -1, (0, 255, 0), 2)
    coner_coordinates = []
    for i in range(len(OPERATION)):
        for j in range(0, 2):
            OPERATION[i][j] = [8 - OPERATION[i][j][0], 8 - OPERATION[i][j][1]]
    for i in contours:
        M = cv2.moments(i)
        chess_board_x = M["m10"]/M["m00"]
        chess_board_y = M["m01"]/M["m00"]
        coner_coordinates.append([chess_board_x, chess_board_y])

BOARD_PHYSICAL_SIZE_MM = 293.0


x_pts = [p[0] for p in coner_coordinates]
y_pts = [p[1] for p in coner_coordinates]

grid_left, grid_right = min(x_pts), max(x_pts)
grid_top, grid_bottom = min(y_pts), max(y_pts)

board_pixel_width = grid_right - grid_left
board_pixel_height = grid_bottom - grid_top


gain_x = BOARD_PHYSICAL_SIZE_MM / board_pixel_width
gain_y = BOARD_PHYSICAL_SIZE_MM / board_pixel_height

cv2.imshow("djalfjldsajflsdlkfjsf", red_mask)






coner_coordinates = sorted(coner_coordinates, key=lambda p: p[1])
top_pts = sorted(coner_coordinates[:2], key=lambda p: p[0])
bottom_pts = sorted(coner_coordinates[2:], key=lambda p: p[0])


src_pts = np.float32([top_pts[0], top_pts[1], bottom_pts[1], bottom_pts[0]])
dst_pts = np.float32(
    [[0.0, 0.0], [293.0, 0.0], [293.0, 293.0], [0.0, 293.0]]
)


H_flatten = cv2.getPerspectiveTransform(src_pts, dst_pts)

H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)






def flatten_point(pixel_x, pixel_y):
    """Converts raw camera pixels into flattened physical board millimeters."""
    pt = np.array([[[pixel_x, pixel_y]]], dtype=np.float32)
    flattened = cv2.perspectiveTransform(pt, H_flatten)
    return flattened[0][0][0], flattened[0][0][1]

def unflatten_point(mm_x, mm_y):
    """Converts physical board millimeters back into camera image pixels."""
    pt = np.array([[[mm_x, mm_y]]], dtype=np.float32)
    unflattened = cv2.perspectiveTransform(pt, H_inv)
    return int(unflattened[0][0][0]), int(unflattened[0][0][1])

CHESS_BOARD_WIDTH = [0.0, 293.0]
CHESS_BOARD_HEIGHT = [0.0, 293.0]


one_blocks_width_x = (CHESS_BOARD_WIDTH[1] - CHESS_BOARD_WIDTH[0]) / 8.0
one_blocks_width_y = (CHESS_BOARD_HEIGHT[1] - CHESS_BOARD_HEIGHT[0]) / 8.0



clawx, clawy = flatten_point(SCREEN_CX, SCREEN_CY)

clawx += 16.221
clawy += 132.192







def capture_board_occupancy():
    """Captures 1 frame, runs YOLO, and returns an 8x8 binary occupancy matrix."""
    
    for _ in range(20):
        mainstream.wait_for_frames()
    frames = mainstream.wait_for_frames()
    aligned = alinger.process(frames)
    frame = np.asanyarray(aligned.get_color_frame().get_data())

    matrix = np.zeros((8, 8), dtype=int)
    results = model(frame, conf=0.3, verbose=False)

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        mm_x, mm_y = flatten_point(cx, cy)

        col = int(mm_x // one_blocks_width_x)
        row = int(mm_y // one_blocks_width_y)

        if 0 <= col < 8 and 0 <= row < 8:
            matrix[row][col] = 1

    return matrix



# def capture_board_occupancy(crop_size=100, conf_thresh=0.07):
#     """Scans all 64 squares using the exact math from the working pick-and-place loop."""
#     matrix = np.zeros((8, 8), dtype=int)
#     scanoperation = []
#     print("--> Scanning board under the claw...")

#     for row in range(8):
#         for col in range(8):
#             scanoperation.append([[row, col], [1,1]])
#     # piece = ["queen"] * len(scanoperation)
#     for i, k in zip(scanoperation, scanoperation):



#         destination_x = CHESS_BOARD_WIDTH[0] + one_blocks_width_x * i[0][0] + one_blocks_width_x/2
#         destination_y =  CHESS_BOARD_HEIGHT[0] + one_blocks_width_y * i[0][1] + one_blocks_width_y/2






#         if k[0][1] < 5:
#             error_x = -1 * (destination_x - clawx + (colum_gain*(-5 + i[0][1])))
#         else:
#             error_x = -1 * (destination_x - clawx + (colum_gain_bottom*(-5 + i[0][1])))
#         error_y = -1 * (destination_y - clawy)

    
#         arm.set_position(HOME_X + error_y, HOME_Y + error_x, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=set_up)
#         time.sleep(0.06)  # Settle vibration

#         # 2. CAPTURE FRESH FRAME AT THIS SPECIFIC SQUARE (INSIDE LOOP)
#         for _ in range(2):
#             mainstream.poll_for_frames()

#         frames = mainstream.wait_for_frames(5000)
#         aligned = alinger.process(frames)
#         color_frame = aligned.get_color_frame()
#         if not color_frame:
#             continue

#         frame = np.asanyarray(color_frame.get_data())

#         # 3. CROP AREA UNDER CAMERA
#         x1 = max(0, SCREEN_CX - crop_size // 2)
#         y1 = max(0, SCREEN_CY - crop_size // 2)
#         x2 = min(frame.shape[1], SCREEN_CX + crop_size // 2)
#         y2 = min(frame.shape[0], SCREEN_CY + crop_size // 2)

#         crop = frame[y1:y2, x1:x2]
#         if crop.size == 0:
#             continue

#         # 4. IDENTIFY PIECE FOR THIS SQUARE
#         results = model(crop, conf=conf_thresh, verbose=False)
#         if len(results[0].boxes) > 0:
#             matrix[i[0][0]][i[0][1]] = 1
#         else:
#             # Fallback color thresholding
#             hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#             white_mask = cv2.inRange(
#                 hsv, np.array([0, 20, 140]), np.array([180, 50, 255])
#             )
#             dark_mask = cv2.inRange(
#                 hsv, np.array([0, 0, 0]), np.array([180, 255, 60])
#             )

#             if (
#                 cv2.countNonZero(white_mask) > 40
#                 or cv2.countNonZero(dark_mask) > 50
#             ):
#                 matrix[i[0][0]][i[0][1]] = 1
        
            

#     # Return arm to HOME baseline
#     arm.set_position(
#         HOME_X,
#         HOME_Y,
#         HOME_Z,
#         HOME_ROLL,
#         HOME_PITCH,
#         HOME_YAW,
#         SPEED,
#         wait=True,
#     )

#     return matrix
previous_matrix = np.array(
    [
        [1, 1, 1, 1, 1, 1, 1, 1],  # Row 0 (Rank 8)
        [1, 1, 1, 1, 1, 1, 1, 1],  # Row 1 (Rank 7)
        [0, 0, 0, 0, 0, 0, 0, 0],  # Row 2 (Rank 6)
        [0, 0, 0, 0, 0, 0, 0, 0],  # Row 3 (Rank 5)
        [0, 0, 0, 0, 0, 0, 0, 0],  # Row 4 (Rank 4)
        [0, 0, 0, 0, 0, 0, 0, 0],  # Row 5 (Rank 3)
        [1, 1, 1, 1, 1, 1, 1, 1],  # Row 6 (Rank 2)
        [1, 1, 1, 1, 1, 1, 1, 1],  # Row 7 (Rank 1)
    ],
    dtype=int,
)

def grid_to_uci(col, row):

    files = ["a", "b", "c", "d", "e", "f", "g", "h"]
    ranks = ["8", "7", "6", "5", "4", "3", "2", "1"]
    return f"{files[col]}{ranks[row]}"


def uci_to_grid(uci_str):
  
    files = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
    ranks = {"8": 0, "7": 1, "6": 2, "5": 3, "4": 4, "3": 5, "2": 6, "1": 7}
    return [files[uci_str[0]], ranks[uci_str[1]]]
def get_closest_yolo_piece_mm(target_mm_x, target_mm_y, row, col, conf_thresh=0.08, max_search_radius_mm=50.0):
    
 
    frames = mainstream.wait_for_frames(5000)
    aligned = alinger.process(frames)
    color_frame = aligned.get_color_frame()
    if not color_frame:
        return target_mm_x, target_mm_y

    frame = np.asanyarray(color_frame.get_data())

 
    results = model(frame, conf=conf_thresh, verbose=False)

    closest_point = None
    min_distance = float("inf")


    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

        box_center_px_x = (x1 + x2) / 2.0
        box_center_px_y = (y1 + y2) / 2.0

        mm_x, mm_y = flatten_point(box_center_px_x, box_center_px_y)

        dist = np.hypot(mm_x - target_mm_x, mm_y - target_mm_y)

        if dist < min_distance and dist <= max_search_radius_mm:
            min_distance = dist
            closest_point = [mm_x, mm_y]
    factorrow = (-5 + row)*3.11 * 0
    factorcol = (-5 + col)*3.11 * 0
    

    if closest_point is not None:
        closest_point[0] += factorrow
        closest_point[1] += factorcol
        return closest_point
    


    return target_mm_x, target_mm_y


colum_gain = -1.5
colum_gain_bottom = 2
# colum_gain = 0
# colum_gain_bottom = 0
def is_white_piece_on_square(row, col, block_w=36.6, block_h=36.6):
    
    for _ in range(20):
        mainstream.poll_for_frames()

    frames = mainstream.wait_for_frames(5000)
    aligned = alinger.process(frames)
    color_frame = aligned.get_color_frame()
    if not color_frame:
        return False

    raw_frame = np.asanyarray(color_frame.get_data())

    warped_frame = cv2.warpPerspective(raw_frame, H_flatten, (293, 293))


    x1 = int(col * block_w)
    y1 = int(row * block_h)
    x2 = int((col + 1) * block_w)
    y2 = int((row + 1) * block_h)

    square_crop = warped_frame[y1:y2, x1:x2]
    if square_crop.size == 0:
        return False

    hsv = cv2.cvtColor(square_crop, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 20, 140])
    upper_white = np.array([180, 50, 255])
    white_mask = cv2.inRange(hsv, lower_white, upper_white)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)


    contours, _ = cv2.findContours(
        white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    crop_h, crop_w = square_crop.shape[:2]

    for c in contours:
        if cv2.contourArea(c) > 15:
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"]
                cy = M["m01"] / M["m00"]


                if (0.2 * crop_w) <= cx <= (0.8 * crop_w) and (
                    0.2 * crop_h
                ) <= cy <= (0.8 * crop_h):
                    return True

    return False


import cv2
import numpy as np





print(CHESS_BOARD_WIDTH, CHESS_BOARD_HEIGHT)

def display():
    for i in range(3000000000000000):
        frames = mainstream.wait_for_frames()
        aligned_frames = alinger.process(frames)
        live_color_data = np.asanyarray(
            aligned_frames.get_color_frame().get_data()
        )

        output_view = live_color_data.copy()

        claw_px_x, claw_px_y = unflatten_point(clawx, clawy)
        target_px_x, target_px_y = unflatten_point(
            destination_x, destination_y
        )

    
        cv2.circle(output_view, (claw_px_x, claw_px_y), 8, (0, 0, 255), -1)
        cv2.putText(
            output_view,
            "CLAW",
            (claw_px_x + 10, claw_px_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

  
        cv2.circle(output_view, (target_px_x, target_px_y), 8, (0, 255, 0), -1)
        cv2.putText(
            output_view,
            f"TARGET {origninal}",
            (target_px_x + 10, target_px_y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

      
        cv2.imshow("Live Continuous Vision", output_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("--> Stream stopped manually.")
            break
        time.sleep(0.00000001)


current_matrix = previous_matrix.copy()


def get_white_pieces_matrix():
    """Captures a frame, warps it to top-down view, and returns

    an 8x8 matrix where 1 = White Piece, 0 = Empty/Black Piece.
    """
    # 1. Capture and align frame
    frames = mainstream.wait_for_frames(5000)
    aligned = alinger.process(frames)
    frame = np.asanyarray(aligned.get_color_frame().get_data())

    # 2. Warp image to flat board space (293x293 px)
    warped = cv2.warpPerspective(frame, H_flatten, (293, 293))

    block_w = 293.0 / 8.0
    block_h = 293.0 / 8.0

    white_matrix = np.zeros((8, 8), dtype=int)

    # 3. Check every square for a white piece
    for row in range(8):
        for col in range(8):
            arm.set_position(HOME_X+10, HOME_Y, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
            if is_white_piece_on_square(row, col):
                white_matrix[row][col] = 1
                
            arm.set_position(HOME_X-10, HOME_Y, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
            if is_white_piece_on_square(row, col):
                white_matrix[row][col] = 1
            arm.set_position(HOME_X, HOME_Y-10, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
            if is_white_piece_on_square(row, col):
                white_matrix[row][col] = 1
            arm.set_position(HOME_X, HOME_Y+10, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
            if is_white_piece_on_square(row, col):
                white_matrix[row][col] = 1
            arm.set_position(HOME_X, HOME_Y, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
            if is_white_piece_on_square(row, col):
                white_matrix[row][col] = 1
            

    return white_matrix


# Print the current White piece matrix
# white_board = get_white_pieces_matrix()
# print("White Pieces Matrix (1 = White, 0 = Empty/Black):")
# print(white_board)
# exit()

try:
    while not board_logic.is_game_over():

        # ----------------------------------------------------------------------
        # STEP 1: WAIT FOR WHITE (HUMAN) MOVE
        # ----------------------------------------------------------------------
        print(current_matrix)
        # print(get_white_pieces_matrix())
        input("\n[White's Turn] Make your move and press ENTER...")

        matrix = np.zeros((8, 8), dtype=int)
        mylist = []
        current_matrix = np.zeros((8, 8), dtype=int)
        
    
        arm.set_position(HOME_X+5, HOME_Y, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
        for i in range(2):
            mylist.append(capture_board_occupancy())
        arm.set_position(HOME_X, HOME_Y+5, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
        for i in range(2):
            mylist.append(capture_board_occupancy())
        arm.set_position(HOME_X-5, HOME_Y, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
        for i in range(2):
            mylist.append(capture_board_occupancy())
        arm.set_position(HOME_X, HOME_Y-5, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
        for i in range(2):
            mylist.append(capture_board_occupancy())
        arm.set_position(HOME_X, HOME_Y, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
        for i in range(2):
            mylist.append(capture_board_occupancy())


        
        counter = 0
        for i in range(8):
            for j in range(8):
                counter = 0
                for k in mylist:
                    if k[i][j] == 1:
                        counter += 1
                if counter > 0:
                    current_matrix[i][j] = 1




        # current_matrix = capture_board_occupancy()

        # Compare matrices to find origin (1 -> 0) and destination (0 -> 1)
        from_col_row = None
        to_col_row = None

        for r in range(8):
            for c in range(8):
                if previous_matrix[r][c] == 1 and current_matrix[r][c] == 0:
                    from_col_row = (c, r)
                elif previous_matrix[r][c] == 0 and current_matrix[r][c] == 1:
                    to_col_row = (c, r)
        # Helper to check the center of the camera view under the claw
        def check_white_under_claw(crop_size=80, white_pixel_thresh=50):
            """Checks if a white piece is centered directly under the camera/claw."""
            for _ in range(3):
                mainstream.poll_for_frames()

            frames = mainstream.wait_for_frames(5000)
            aligned = alinger.process(frames)
            frame = np.asanyarray(aligned.get_color_frame().get_data())

            # Crop the central area directly beneath the camera
            x1 = max(0, SCREEN_CX - crop_size // 2)
            y1 = max(0, SCREEN_CY - crop_size // 2)
            x2 = min(frame.shape[1], SCREEN_CX + crop_size // 2)
            y2 = min(frame.shape[0], SCREEN_CY + crop_size // 2)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                return False

            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 20, 170])
            upper_white = np.array([180, 50, 255])
            white_mask = cv2.inRange(hsv, lower_white, upper_white)

            return cv2.countNonZero(white_mask) > white_pixel_thresh


        # --- Fallback Legal Move Search Loop ---
        if to_col_row is None and from_col_row is not None:
            from_sq = chess.square(from_col_row[0], 7 - from_col_row[1])
            legal_moves = list(board_logic.generate_legal_moves(from_mask=1 << from_sq))

            print(f"--> Searching {len(legal_moves)} legal target squares under the claw...")
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

            color_sensor.set_option(rs.option.exposure, 170.0)
            color_sensor.set_option(rs.option.white_balance, 3500.0)
            for _ in range(20):
                mainstream.poll_for_frames()

            # # Lower gain slightly to reduce sensor noise
            color_sensor.set_option(rs.option.gain, 50.0)
            for move in legal_moves:
                target_sq = move.to_square
                col = chess.square_file(target_sq)  # 0 to 7
                row = 7 - chess.square_rank(target_sq)  # 0 to 7

                # 1. Calculate physical millimeter destination for this square
                # dest_x = CHESS_BOARD_WIDTH[0] + one_blocks_width_x * col + one_blocks_width_x / 2.0
                # dest_y = CHESS_BOARD_HEIGHT[0] + one_blocks_width_y * row + one_blocks_width_y / 2.0

                # # 2. Compute arm offset errors
                # gain = colum_gain if row < 5 else colum_gain_bottom
                # err_x = -1.0 * (dest_x - clawx + (gain * (-5 + row)))
                # err_y = -1.0 * (dest_y - clawy)
                print("here comes the hard part..")
                if is_white_piece_on_square(row, col):
                    to_col_row = (col, row)

            #     # 3. Move arm directly over the square
            #     arm.set_position(
            #         HOME_X + err_y,
            #         HOME_Y + err_x,
            #         HOME_Z,
            #         HOME_ROLL,
            #         HOME_PITCH,
            #         HOME_YAW,
            #         SPEED,
            #         wait=True,
            #     )
            #     time.sleep(0.12)  # Settle arm vibration

            #     # 4. Check for white piece directly under the camera
            #     if check_white_under_claw():
            #         to_col_row = (col, row)
            #         print(f"--> Found piece at square: {grid_to_uci(col, row)} ({col}, {row})")
            #         break

            # # 5. Always return arm to HOME position before continuing
            # arm.set_position(
            #     HOME_X,
            #     HOME_Y,
            #     HOME_Z,
            #     HOME_ROLL,
            #     HOME_PITCH,
            #     HOME_YAW,
            #     SPEED,
            #     wait=True,
            # )
        # if to_col_row is None and from_col_row != None:
        #     for _ in range(30):
        #         mainstream.wait_for_frames()

        #     frames = mainstream.wait_for_frames(50000)
        #     aligned = alinger.process(frames)
        #     frame = np.asanyarray(aligned.get_color_frame().get_data())
        #     warped_frame = cv2.warpPerspective(frame, H_flatten, (293, 293))
        #     from_sq = chess.square(from_col_row[0], 7 - from_col_row[1])
        #     legal_moves = list(
        #         board_logic.generate_legal_moves(from_mask=1 << from_sq)
        #     )
        #     for i in legal_moves:
        #         found = False
        #         target_sq = i.to_square  # Returns an integer (0 to 63)

        #         # 1. Convert to 2D matrix (row, col) coordinates
        #         col = chess.square_file(target_sq)  # File a-h -> 0 to 7
        #         row = 7 - chess.square_rank(target_sq)  # Rank 1-8 -> 7 to 0 (Rank 8 is row 0)

        #         destination_x = CHESS_BOARD_WIDTH[0] + one_blocks_width_x * col + one_blocks_width_x/2
        #         destination_y =  CHESS_BOARD_HEIGHT[0] + one_blocks_width_y * row + one_blocks_width_y/2
        
        

                
        #         error_x = -1 * (destination_x - clawx)
        #         error_y = -1 * (destination_y - clawy)

        #         arm.set_position(HOME_X + error_y, HOME_Y + error_x, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=set_up)
        #         if is_white_piece_on_square(row, col):
        #             to_col_row = (col, row)
        #             break
                # if is_white_piece_on_square(row, col):
                #     to_col_row = (col, row)
                    
                # arm.set_position(HOME_X-5, HOME_Y, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
                # color_sensor.set_option(rs.option.exposure, 150.0)
                # if is_white_piece_on_square(row, col):
                #     to_col_row = (col, row)
                #     break
                # arm.set_position(HOME_X, HOME_Y-5, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
                # color_sensor.set_option(rs.option.exposure, 200.0)
                # if is_white_piece_on_square(row, col):
                #     to_col_row = (col, row)
                #     break
                # arm.set_position(HOME_X, HOME_Y+5, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
                # color_sensor.set_option(rs.option.exposure, 150.0)
                # if is_white_piece_on_square(row, col):
                #     to_col_row = (col, row)
                #     break
                # arm.set_position(HOME_X, HOME_Y, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=True)
                # color_sensor.set_option(rs.option.exposure, 150.0)
                # if is_white_piece_on_square(row, col):
                #     to_col_row = (col, row)
                #     break
                        


        for _ in range(20):
                mainstream.poll_for_frames()
        color_sensor.set_option(rs.option.enable_auto_exposure, 0)
        color_sensor.set_option(rs.option.enable_auto_white_balance, 1)
        color_sensor.set_option(rs.option.exposure, 250.0)
        
        # color_sensor.set_option(rs.option.exposure, 200.0)
        if from_col_row is None or to_col_row is None:
            print("⚠️ Board change unclear! Ensure pieces sit centered on squares.")
            continue

        # Format move to UCI string (e.g., "e2e4")
        white_move_uci = grid_to_uci(
            from_col_row[0], from_col_row[1]
        ) + grid_to_uci(to_col_row[0], to_col_row[1])
        print(f"--> Detected Opponent Move: {white_move_uci}")

        # Validate move against python-chess rules
        move_obj = chess.Move.from_uci(white_move_uci)
        if move_obj in board_logic.legal_moves:
            board_logic.push(move_obj)
            previous_matrix = current_matrix.copy()
        else:
            print(f"❌ Illegal move detected ({white_move_uci}). Please fix board setup!")
            continue

        # ----------------------------------------------------------------------
        # STEP 2: CONSULT STOCKFISH FOR BLACK (ROBOT) MOVE
        # ----------------------------------------------------------------------
        print("--> Stockfish calculating response for Black...")
        result = engine.play(board_logic, chess.engine.Limit(time=1.0))
        black_move_uci = result.move.uci()
            
        print(f"--> Stockfish Selected Move for Black: {black_move_uci}")


        

        start_grid = uci_to_grid(black_move_uci[:2])
        end_grid = uci_to_grid(black_move_uci[2:4])

        start_square_idx = chess.parse_square(black_move_uci[:2])
        end_square_idx = chess.parse_square(black_move_uci[2:4])

        # 1. Look up pieces BEFORE pushing the move
        moving_piece = board_logic.piece_at(start_square_idx)
        captured_piece = board_logic.piece_at(end_square_idx)

        # Safe piece name resolution
        moving_name = chess.piece_name(moving_piece.piece_type) if moving_piece else "pawn"

        # 2. Advance the logic tracker
        board_logic.push(result.move)

        # 3. Build the OPERATION queue
        
        OPERATION = []
        piece = []
        if black_move_uci == "e8g8":
            print("🏰 Hardcoded Castling: Black Kingside (King e8->g8, Rook h8->f8)")
            OPERATION.append([[4, 0], [6, 0]])
            piece.append("king")
            OPERATION.append([[7, 0], [5, 0]])
            piece.append("rook")

            previous_matrix[0][4] = 0
            previous_matrix[0][6] = 1
            previous_matrix[0][7] = 0
            previous_matrix[0][5] = 1

        # 2. Hardcoded Black Queenside Castling
        elif black_move_uci == "e8c8":
            print("🏰 Hardcoded Castling: Black Queenside (King e8->c8, Rook a8->d8)")
            OPERATION.append([[4, 0], [2, 0]])
            piece.append("king")
            OPERATION.append([[0, 0], [3, 0]])
            piece.append("rook")

            previous_matrix[0][4] = 0
            previous_matrix[0][2] = 1
            previous_matrix[0][0] = 0
            previous_matrix[0][3] = 1

        # 3. Standard Move or Capture
        else:
            if captured_piece is not None:
                captured_name = chess.piece_name(captured_piece.piece_type)
                print(f"⚔️ Capture detected: Removing {captured_name} at {end_grid} to graveyard [9, 5]")
                OPERATION.append([end_grid, [9, 5]])
                piece.append(captured_name)

            OPERATION.append([start_grid, end_grid])
            piece.append(moving_name)

            previous_matrix[start_grid[1]][start_grid[0]] = 0
            previous_matrix[end_grid[1]][end_grid[0]] = 1
        
        print(f"--> Generated OPERATION for xArm: {OPERATION}")

        # If it's a capture, take out the piece first to [12, 0]
        # if captured_piece is not None:
        #     captured_name = chess.piece_name(captured_piece.piece_type)
        #     print(f"⚔️ Capture detected: Removing {captured_name} at {end_grid} to graveyard [12, 0]")
        #     OPERATION.append([end_grid, [9, 5]])
        #     piece.append(captured_name)

        # Add the actual robot move
        # OPERATION.append([start_grid, end_grid])
        # piece.append(moving_name)
        
        print(f"--> Generated OPERATION for xArm: {OPERATION}")

        for i, j, k in zip(OPERATION, piece, OPERATION):



            destination_x = CHESS_BOARD_WIDTH[0] + one_blocks_width_x * i[0][0] + one_blocks_width_x/2
            destination_y =  CHESS_BOARD_HEIGHT[0] + one_blocks_width_y * i[0][1] + one_blocks_width_y/2






            if k[0][1] < 5:
                error_x = -1 * (destination_x - clawx + (colum_gain*(-5 + i[0][1])))
            else:
                error_x = -1 * (destination_x - clawx + (colum_gain_bottom*(-5 + i[0][1])))
            error_y = -1 * (destination_y - clawy)

        
            arm.set_position(HOME_X + error_y, HOME_Y + error_x, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=set_up)
            claw_px_x, claw_px_y = unflatten_point(clawx, clawy)
            destination_x, destination_y = get_closest_yolo_piece_mm(clawx, clawy, i[0][0], i[0][1])
            finederror_x = -1 * (destination_x - clawx)
            finederror_y = -1 * (destination_y - clawy)
            arm.set_position(HOME_X+ finederror_y+error_y, HOME_Y +finederror_x+ error_x, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=set_up)
                


            arm.set_position(HOME_X + error_y, HOME_Y + error_x,  pieces[j], HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=set_up)
            
            close_claw()
            arm.set_position(HOME_X + error_y, HOME_Y + error_x, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=set_up)

            
            destination_x = CHESS_BOARD_WIDTH[0] + one_blocks_width_x * i[1][0] + one_blocks_width_x/2
            destination_y =  CHESS_BOARD_HEIGHT[0] + one_blocks_width_y * i[1][1] + one_blocks_width_y/2
            
            

            if k[0][1] < 5:
                error_x = -1 * (destination_x - clawx + (colum_gain*(-5 + i[0][1])))
            else:
                error_x = -1 * (destination_x - clawx + (colum_gain_bottom*(-5 + i[0][1])))
            error_y = -1 * (destination_y - clawy)

            arm.set_position(HOME_X + error_y, HOME_Y + error_x, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=set_up)
            
            arm.set_position(HOME_X + error_y, HOME_Y + error_x,  pieces[j]+6, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=set_up)

            open_claw()
            arm.set_position(HOME_X, HOME_Y, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=set_up)





























        previous_matrix[start_grid[1]][start_grid[0]] = 0
        previous_matrix[end_grid[1]][end_grid[0]] = 1

finally:
    engine.quit()
    mainstream.stop()
    cv2.destroyAllWindows()
