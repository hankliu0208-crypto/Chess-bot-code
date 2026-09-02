import os
import random
import shutil
from ultralytics import YOLO


RAW_DIR = os.path.abspath("white_piece_dataset")
DATASET_DIR = os.path.abspath("white_detection")

if not os.path.exists(RAW_DIR):
    print(f"Error: '{RAW_DIR}' folder not found! Capture photos first.")
    exit()


for split in ["train", "val"]:
    os.makedirs(os.path.join(DATASET_DIR, "images", split), exist_ok=True)
    os.makedirs(os.path.join(DATASET_DIR, "labels", split), exist_ok=True)


images = [
    f
    for f in os.listdir(RAW_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not images:
    print(f"No images found in '{RAW_DIR}'!")
    exit()

random.shuffle(images)
split_idx = max(1, int(len(images) * 0.8))
train_imgs = images[:split_idx]
val_imgs = images[split_idx:] if len(images) > 1 else train_imgs


def copy_files(img_list, split):
    for img_name in img_list:
        base_name = os.path.splitext(img_name)[0]
        txt_name = base_name + ".txt"

        src_img = os.path.join(RAW_DIR, img_name)
        src_txt = os.path.join(RAW_DIR, txt_name)

        dst_img = os.path.join(DATASET_DIR, "images", split, img_name)
        dst_txt = os.path.join(DATASET_DIR, "labels", split, txt_name)

        shutil.copy(src_img, dst_img)
        if os.path.exists(src_txt):
            shutil.copy(src_txt, dst_txt)


copy_files(train_imgs, "train")
copy_files(val_imgs, "val")


yaml_path = os.path.join(DATASET_DIR, "data.yaml")
yaml_content = f"""path: {DATASET_DIR}
train: images/train
val: images/val

names:
  0: piece
"""

with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"Dataset successfully prepared at: {yaml_path}")

model = YOLO("yolov8n.pt")
model.train(
    data=yaml_path,
    epochs=50,
    imgsz=640,
    batch=16,
    name="chess_piece_detector",
    workers=2,
)

print("\nTraining complete!")
print(
    "Weights saved to: runs/detect/chess_piece_detector/weights/best.pt"
)