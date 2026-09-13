import os
import shutil
import random

# Paths
LION_DATASET = r"C:\WildGuard\raw\lion_extra"
DATASET_DIR = r"C:\WildGuard\dataset"

# Lion class index in YOUR existing dataset
# Check dataset/classes.txt — what number is lion?
LION_CLASS_IDX = 7  # lion is index 7 in your wildguard.yaml

random.seed(42)
MAX_PER_SPLIT = {"train": 350, "val": 75, "test": 75}

for split in ["train", "valid", "test"]:
    # Roboflow uses "valid" not "val"
    src_img = os.path.join(LION_DATASET, split, "images")
    src_lbl = os.path.join(LION_DATASET, split, "labels")
    
    # Map roboflow split name to your split name
    dst_split = "val" if split == "valid" else split
    dst_img = os.path.join(DATASET_DIR, dst_split, "images")
    dst_lbl = os.path.join(DATASET_DIR, dst_split, "labels")

    if not os.path.exists(src_img):
        print(f"Skipping {split} — not found")
        continue

    images = [f for f in os.listdir(src_img) if f.endswith(('.jpg', '.png'))]
    random.shuffle(images)
    images = images[:MAX_PER_SPLIT[dst_split]]

    copied = 0
    for img_file in images:
        # Copy image
        src_i = os.path.join(src_img, img_file)
        dst_i = os.path.join(dst_img, f"lion_extra_{img_file}")
        shutil.copy2(src_i, dst_i)

        # Copy and fix label
        lbl_file = img_file.rsplit('.', 1)[0] + '.txt'
        src_l = os.path.join(src_lbl, lbl_file)
        dst_l = os.path.join(dst_lbl, f"lion_extra_{lbl_file}")

        if os.path.exists(src_l):
            with open(src_l) as f:
                lines = f.readlines()
            with open(dst_l, 'w') as f:
                for line in lines:
                    parts = line.strip().split()
                    if parts:
                        # Override class index to your lion class
                        parts[0] = str(LION_CLASS_IDX)
                        f.write(' '.join(parts) + '\n')
            copied += 1

    print(f"{dst_split}: copied {copied} lion images")

print("Done — lion dataset merged!")