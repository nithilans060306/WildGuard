import json

with open(r"C:\WildGuard\raw\annotations_coco.json") as f:
    data = json.load(f)

# Remove human category
data["categories"] = [c for c in data["categories"] if c["name"] != "human"]

# Get valid category ids
valid_ids = {c["id"] for c in data["categories"]}

# Remove human annotations
data["annotations"] = [a for a in data["annotations"] if a["category_id"] in valid_ids]

# Remove human images (those with only human annotations)
valid_img_ids = {a["image_id"] for a in data["annotations"]}
data["images"] = [img for img in data["images"] if img["id"] in valid_img_ids]

with open(r"C:\WildGuard\raw\annotations_coco.json", "w") as f:
    json.dump(data, f, indent=2)

print("Done")
print("Remaining images:", len(data["images"]))
print("Remaining annotations:", len(data["annotations"]))
print("Remaining classes:", [c["name"] for c in data["categories"]])