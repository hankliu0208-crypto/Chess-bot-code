import cv2
import numpy as np
import pyrealsense2 as rs


pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)



profile = pipeline.start(config)  


color_sensor = profile.get_device().first_color_sensor()


color_sensor.set_option(rs.option.enable_auto_exposure, 0)
color_sensor.set_option(rs.option.enable_auto_white_balance, 0)

color_sensor.set_option(rs.option.exposure, 170.0)
color_sensor.set_option(rs.option.white_balance, 3500.0)


color_sensor.set_option(rs.option.gain, 50.0)

lower_white = np.array([0, 0, 0])
upper_white = np.array([180, 35, 255])

lower_white = np.array([0, 20, 170])
upper_white = np.array([180, 50, 255])

print(
    "--> Showing clean feed vs white mask. Press 'q' in the window to exit."
)

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        frame = np.asanyarray(color_frame.get_data())
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


        raw_mask = cv2.inRange(hsv, lower_white, upper_white) 
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        clean_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)


        mask_bgr = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)

        combined_view = np.hstack((frame, mask_bgr))

        cv2.imshow(
            "W",
            combined_view,
        )

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()