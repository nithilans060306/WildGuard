import json

with open("raw/annotations_coco.json") as f:
    data = json.load(f)

print("Total images:", len(data["images"]))
print("Total annotations:", len(data["annotations"]))
print("\nCategories:")
for cat in data["categories"]:
    print(" ", cat)

print("\nSample annotation:", data["annotations"][0])

# Count per class
from collections import Counter
cat_ids = [a["category_id"] for a in data["annotations"]]
counts = Counter(cat_ids)
print("\nAnnotations per category_id:", counts)

import json, zipfile
from collections import Counter

with zipfile.ZipFile("wcs_20220205_bboxes_with_classes.zip", "r") as z:
    json_files = [n for n in z.namelist() if n.endswith(".json")]
    with z.open(json_files[0]) as f:
        data = json.load(f)

# Map id to name
id_to_name = {c["id"]: c["name"] for c in data["categories"]}

# Count annotations per category
counts = Counter(a["category_id"] for a in data["annotations"])

# Classes we want
target_ids = [
    100, 154, 122, 104, 24,       # big cats
    98, 265, 293, 390, 153,        # other cats (caracal, serval, clouded leopard, marbled cat, temminck)
    90, 97, 163, 261, 260,         # elephant, hyena, wild dog, white rhino, black rhino
    268, 110, 287,                  # hippo, buffalo, wildebeest
    319, 78, 437,                   # chimp, baboon, yellow baboon
    75, 676, 558,                   # human, vehicle, motorcycle
    620, 252
]

print("Class availability check:")
for cid in target_ids:
    name = id_to_name.get(cid, "NOT FOUND")
    count = counts.get(cid, 0)
    print(f"  id={cid:4d}  {name:35s}  annotations={count:,}")