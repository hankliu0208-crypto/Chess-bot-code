import os
import cv2
from ultralytics import YOLO

IMAGE_DIR = "white_piece_dataset"
os.makedirs(IMAGE_DIR, exist_ok=True)


MODEL_PATH = "runs/detect/chess_piece_detector-5/weights/best.pt"
CONF_THRESH = 0.10  

print(f"Loading YOLO model from '{MODEL_PATH}'...")
model = YOLO(MODEL_PATH)

images = sorted(
    [
        f
        for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
)

if not images:
    print(f"No images found in '{IMAGE_DIR}'. Capture photos first!")
    exit()

boxes = []
drawing = False
ix, iy = -1, -1
current_box = None


def mouse_draw_callback(event, x, y, flags, param):
    global ix, iy, drawing, current_box, boxes

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        current_box = (x, y, x, y)

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            current_box = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = min(ix, x), min(iy, y)
        x2, y2 = max(ix, x), max(iy, y)
        if (x2 - x1) > 5 and (y2 - y1) > 5:
            boxes.append((x1, y1, x2, y2))
        current_box = None


cv2.namedWindow("Mac Custom Labeler")
cv2.setMouseCallback("Mac Custom Labeler", mouse_draw_callback)

for img_name in images:
    txt_name = os.path.splitext(img_name)[0] + ".txt"
    txt_path = os.path.join(IMAGE_DIR, txt_name)

    # SKIP ALREADY LABELED IMAGES
    if os.path.exists(txt_path) and os.path.getsize(txt_path) > 0:
        print(f"Skipping already labeled image: {img_name}")
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    img_h, img_w, _ = img.shape
    boxes = []

    while True:
        display_img = img.copy()

        # Draw confirmed bounding boxes
        for b in boxes:
            cv2.rectangle(
                display_img, (b[0], b[1]), (b[2], b[3]), (0, 255, 0), 2
            )

        # Draw active drag box
        if current_box is not None:
            cv2.rectangle(
                display_img,
                (current_box[0], current_box[1]),
                (current_box[2], current_box[3]),
                (255, 0, 0),
                1,
            )

        hud = f"{img_name} | Boxes: {len(boxes)} | 'h': AI Assist | 'u': Undo | ENTER: Save | 'q': Quit"
        cv2.putText(
            display_img,
            hud,
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
        )

        cv2.imshow("Mac Custom Labeler", display_img)
        key = cv2.waitKey(20) & 0xFF

        # Auto-detect using AI
        if key == ord("h"):
            results = model(img, conf=CONF_THRESH, verbose=False)
            ai_count = 0
            for r_box in results[0].boxes:
                x1, y1, x2, y2 = r_box.xyxy[0].cpu().numpy().astype(int)
                # Avoid adding identical overlapping duplicate boxes
                if not any(
                    abs(b[0] - x1) < 10 and abs(b[1] - y1) < 10 for b in boxes
                ):
                    boxes.append((x1, y1, x2, y2))
                    ai_count += 1
            print(f"--> AI generated {ai_count} bounding boxes.")

        elif key == ord("u") and boxes:
            boxes.pop()

        elif key in (13, 10, ord("n")):
            break

        elif key == ord("q"):
            cv2.destroyAllWindows()
            exit()

    # Save final YOLO normalized labels
    with open(txt_path, "w") as f:
        for b in boxes:
            x1, y1, x2, y2 = b
            box_w = x2 - x1
            box_h = y2 - y1
            x_center = (x1 + box_w / 2.0) / img_w
            y_center = (y1 + box_h / 2.0) / img_h
            norm_w = box_w / img_w
            norm_h = box_h / img_h
            f.write(
                f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}\n"
            )

    print(f"Saved {len(boxes)} piece labels for {img_name}")

cv2.destroyAllWindows()
print("All new images annotated successfully!")