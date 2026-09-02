import time
import cv2
import numpy as np
import pyrealsense2 as rs
from xarm.wrapper import XArmAPI
import threading







XARM_IP = "192.168.1.223"


HOME_X = 290.0  
HOME_Y = 10.0  
HOME_Z = 580.0  

HOME_ROLL = 180
HOME_PITCH = -0.4
HOME_YAW = 2.7
SCREEN_CX, SCREEN_CY = 320, 240
PIXEL_TOLERANCE = 3  






BOARD_PHYSICAL_SIZE_MM = 293.0



LAST_MOVE_TIME = 0.0
MOVE_INTERVAL = 0.20 
CHESS_BOARD_WIDTH = []
CHESS_BOARD_HEIGHT =[]
TOLERACE = 50

SPEED = 60
set_up = False


pieces = {
    "rook": 170,
    "queen": 190,
    "bishop": 170,
    "pawn": 160,
    "knight": 170,
    "king": 190
}






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

# piece = [
#     "rook",
#     "knight",
#     "bishop",
#     "queen",
#     "king",
#     "bishop",
#     "knight",
#     "rook",
# ]



OPERATION = [
    [[4, 1], [8, 2]],
    [[5, 1], [8, 2]],
    [[4, 8], [8, 2]],
    [[5, 8], [8, 6]],
    
    

    ]  # Queen
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
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
mainstream.start(config)
alinger = rs.align(rs.stream.color)
colorizer = rs.colorizer()

for _ in range(20):
    mainstream.wait_for_frames()


frames = mainstream.wait_for_frames(50000)
aligned_frames = alinger.process(frames)
actual_color_data = np.asanyarray(aligned_frames.get_color_frame().get_data())
actual_depth_data = np.asanyarray(aligned_frames.get_depth_frame().get_data())





hsv = cv2.cvtColor(actual_color_data, cv2.COLOR_BGR2HSV)


lower_red1 = np.array([0, 40, 80])
upper_red1 = np.array([12, 255, 255])
mask1 = cv2.inRange(hsv, lower_red1, upper_red1)


lower_red2 = np.array([155, 40, 80])
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









claw_position = [(SCREEN_CX + 17.221/gain_x), SCREEN_CY + 92.192/gain_y]




def get_overhead_peak_pixels(col, row, depth_frame):
    """Finds the highest point (peak depth) of a piece from a direct overhead camera position."""

    x1 = max(0, int(CHESS_BOARD_WIDTH[0] + (col * one_blocks_width_x)))
    x2 = min(
        depth_frame.shape[1],
        int(CHESS_BOARD_WIDTH[0] + ((col + 1) * one_blocks_width_x)),
    )
    y1 = max(0, int(CHESS_BOARD_HEIGHT[0] + (row * one_blocks_width_y)))
    y2 = min(
        depth_frame.shape[0],
        int(CHESS_BOARD_HEIGHT[0] + ((row + 1) * one_blocks_width_y)),
    )

    # Center fallback
    default_x = (
        CHESS_BOARD_WIDTH[0]
        + (col * one_blocks_width_x)
        + (one_blocks_width_x / 2.0)
    )
    default_y = (
        CHESS_BOARD_HEIGHT[0]
        + (row * one_blocks_width_y)
        + (one_blocks_width_y / 2.0)
    )

    depth_roi = depth_frame[y1:y2, x1:x2]
    if depth_roi.size == 0:
        return default_x, default_y

    # 2. Distance sanity limits (Camera at ~550mm height)
    MIN_VALID_DEPTH_MM = 250  # Ignore noise spikes closer than 250mm
    MAX_VALID_DEPTH_MM = 520  # Ignore board floor at ~530mm

    min_depth = MAX_VALID_DEPTH_MM
    best_px_x, best_px_y = default_x, default_y
    found_peak = False

    # 3. Find smallest depth reading (highest point) inside ROI
    for local_y in range(depth_roi.shape[0]):
        for local_x in range(depth_roi.shape[1]):
            val = depth_roi[local_y, local_x]
            if MIN_VALID_DEPTH_MM < val < min_depth:
                min_depth = val
                best_px_x = x1 + local_x
                best_px_y = y1 + local_y
                found_peak = True

    return (best_px_x, best_px_y) if found_peak else (default_x, default_y)

for i in range(len(coner_coordinates)):
    for j in range(len(coner_coordinates)):
        if i != j:
            if abs(coner_coordinates[i][1] -  coner_coordinates[j][1]) < TOLERACE:
                CHESS_BOARD_WIDTH = [min(coner_coordinates[i][0], coner_coordinates[j][0]), max(coner_coordinates[i][0], coner_coordinates[j][0])]
            if abs(coner_coordinates[i][0] - coner_coordinates[j][0]) < TOLERACE:
                CHESS_BOARD_HEIGHT = [min(coner_coordinates[i][1], coner_coordinates[j][1]), max(coner_coordinates[i][1], coner_coordinates[j][1])]





one_blocks_width_x = (CHESS_BOARD_WIDTH[1] - CHESS_BOARD_WIDTH[0]) / 8
one_blocks_width_y = (CHESS_BOARD_HEIGHT[1] - CHESS_BOARD_HEIGHT[0]) / 8




print(CHESS_BOARD_WIDTH, CHESS_BOARD_HEIGHT)

def display():
    for i in range(3000000000000000):
        frames = mainstream.wait_for_frames()
        aligned_frames = alinger.process(frames)
        live_color_data = np.asanyarray(aligned_frames.get_color_frame().get_data())

        

        output_view = live_color_data.copy()

        cv2.circle(output_view, (int(claw_position[0]), int(claw_position[1])), 8, (0, 0, 255), -1)
        cv2.putText(output_view, "CLAW", (int(claw_position[0]) + 10, int(claw_position[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        cv2.circle(output_view, (int(destination_x), int(destination_y)), 8, (0, 255, 0), -1)
        cv2.putText(output_view, f"TARGET {origninal}", (int(destination_x) + 10, int(destination_y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

  
        cv2.imshow("Live C Vision", output_view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("--> Stream stopped manually.")
            break
        time.sleep(0.00000001)

        





for i, j in zip(OPERATION, piece):



    destination_x = CHESS_BOARD_WIDTH[0] + one_blocks_width_x * i[0][0] + one_blocks_width_x/2
    destination_y =  CHESS_BOARD_HEIGHT[0] + one_blocks_width_y * i[0][1] + one_blocks_width_y/2
   

    error_x = -1 * (destination_x - claw_position[0]) * gain_x
    error_y = -1 * (destination_y - claw_position[1]) * gain_y

    display()
    
    arm.set_position(HOME_X + error_y, HOME_Y + error_x, HOME_Z, HOME_ROLL, HOME_PITCH, HOME_YAW, SPEED, wait=set_up)

    

    arm.set_position(
        HOME_X + error_y,
        HOME_Y + error_x,
        HOME_Z,
        HOME_ROLL,
        HOME_PITCH,
        HOME_YAW,
        SPEED,
        wait=set_up,
    )
    display()

    arm.set_position(
        HOME_X + error_y,
        HOME_Y + error_x,
        pieces[j],
        HOME_ROLL,
        HOME_PITCH,
        HOME_YAW,
        SPEED,
        wait=set_up,
    )
   










cv2.waitKey(0)
cv2.destroyAllWindows()


mainstream.stop()












