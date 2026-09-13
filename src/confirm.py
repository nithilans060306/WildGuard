import json

with open(r"C:\WildGuard\raw\annotations_coco.json") as f:
    data = json.load(f)

print("Images:", len(data["images"]))
print("Annotations:", len(data["annotations"]))
print("Classes:", [c["name"] for c in data["categories"]])