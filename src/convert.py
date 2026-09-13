import json
import os
import shutil
import random
from collections import defaultdict

# ─── CONFIG ───────────────────────────────────────────────
RAW_IMAGES_DIR = r"C:\WildGuard\raw\images"
ANNOTATIONS    = r"C:\WildGuard\raw\annotations_coco.json"
OUTPUT_DIR     = r"C:\WildGuard\dataset"

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15

random.seed(42)
# ──────────────────────────────────────────────────────────

# Load annotations
with open(ANNOTATIONS) as f:
    data = json.load(f)

# Build lookups
id_to_image = {img["id"]: img for img in data["images"]}
cat_id_to_idx = {cat["id"]: i for i, cat in enumerate(data["categories"])}
class_names = [cat["name"] for cat in data["categories"]]

# Group annotations by image_id
img_to_anns = defaultdict(list)
for ann in data["annotations"]:
    img_to_anns[ann["image_id"]].append(ann)

# Only keep images that have annotations and exist on disk
valid_image_ids = []
for img_id, img in id_to_image.items():
    src_path = os.path.join(RAW_IMAGES_DIR, img["file_name"].replace("/", os.sep))
    if os.path.exists(src_path) and img_id in img_to_anns:
        valid_image_ids.append(img_id)

print(f"Valid images with annotations: {len(valid_image_ids)}")

# Shuffle and split
random.shuffle(valid_image_ids)
total = len(valid_image_ids)
train_end = int(total * TRAIN_RATIO)
val_end   = train_end + int(total * VAL_RATIO)

splits = {
    "train": valid_image_ids[:train_end],
    "val":   valid_image_ids[train_end:val_end],
    "test":  valid_image_ids[val_end:]
}

print(f"Train: {len(splits['train'])}  Val: {len(splits['val'])}  Test: {len(splits['test'])}")

# Create output folders
for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(OUTPUT_DIR, split, "images"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, split, "labels"), exist_ok=True)

# Convert and copy
def convert_bbox(bbox, img_w, img_h):
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    nw = w / img_w
    nh = h / img_h
    return cx, cy, nw, nh

skipped = 0
for split, img_ids in splits.items():
    for img_id in img_ids:
        img      = id_to_image[img_id]
        src_path = os.path.join(RAW_IMAGES_DIR, img["file_name"].replace("/", os.sep))
        fname    = img["wcs_id"] + ".jpg"

        # Copy image
        dst_img = os.path.join(OUTPUT_DIR, split, "images", fname)
        shutil.copy2(src_path, dst_img)

        # Write YOLO label
        dst_lbl = os.path.join(OUTPUT_DIR, split, "labels", fname.replace(".jpg", ".txt"))
        with open(dst_lbl, "w") as lf:
            for ann in img_to_anns[img_id]:
                cls_idx = cat_id_to_idx[ann["category_id"]]
                cx, cy, nw, nh = convert_bbox(ann["bbox"], img["width"], img["height"])
                # skip bad boxes
                if nw <= 0 or nh <= 0 or cx > 1 or cy > 1:
                    skipped += 1
                    continue
                lf.write(f"{cls_idx} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

print(f"Done. Skipped {skipped} bad boxes.")
print(f"Dataset ready at: {OUTPUT_DIR}")

# Save class names for reference
with open(os.path.join(OUTPUT_DIR, "classes.txt"), "w") as f:
    for name in class_names:
        f.write(name + "\n")

print("Classes saved to dataset/classes.txt")