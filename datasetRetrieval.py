import os
import json
import zipfile
import random
import requests
from pathlib import Path
from urllib.parse import quote
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except ImportError:
    print("Installing tqdm...")
    os.system("pip install tqdm")
    from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Your WCS annotation ZIP file
ANNOTATION_ZIP = "wcs_20220205_bboxes_with_classes.zip"

# Output directory
OUTPUT_DIR = Path("raw")

# Images will be stored here
IMAGE_DIR = OUTPUT_DIR / "images"

# Maximum number of images per class
MAX_IMAGES_PER_CLASS = 600

# Number of simultaneous downloads
MAX_WORKERS = 8

# Random seed for reproducibility
RANDOM_SEED = 42

# WCS image storage
BASE_URL = (
    "https://storage.googleapis.com/"
    "public-datasets-lila/wcs-unzipped/"
)

# ------------------------------------------------------------
# Classes selected for WildGuard
# ------------------------------------------------------------

TARGET_CLASSES = {
    "loxodonta africana": "elephant",
    "panthera onca": "jaguar",
    "crocuta crocuta": "hyena",
    "syncerus caffer": "buffalo",
    "papio anubis": "baboon",
    "panthera pardus": "leopard",
    "panthera tigris": "tiger",
    "panthera leo": "lion",
    "pan troglodytes": "chimp",
    "pardofelis temminckii": "golden_cat",
    "neofelis nebulosa": "clouded_leopard",
    "vehicle": "vehicle",
}


# ============================================================
# 2. CREATE OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(exist_ok=True)
IMAGE_DIR.mkdir(exist_ok=True)


# ============================================================
# 3. CHECK ANNOTATION ZIP
# ============================================================

if not os.path.exists(ANNOTATION_ZIP):
    print("\nERROR:")
    print(f"Could not find: {ANNOTATION_ZIP}")
    print("\nMake sure datasetRetrival.py is inside the WildGuard folder.")
    input("\nPress Enter to exit...")
    raise SystemExit


print("=" * 70)
print("WILDGUARD DATASET RETRIEVAL")
print("=" * 70)

print("\nAnnotation file found:")
print(f"  {ANNOTATION_ZIP}")


# ============================================================
# 4. READ JSON FROM ZIP
# ============================================================

print("\nReading WCS annotations...")

with zipfile.ZipFile(ANNOTATION_ZIP, "r") as z:

    json_files = [
        name for name in z.namelist()
        if name.lower().endswith(".json")
    ]

    if not json_files:
        raise RuntimeError(
            "No JSON annotation file found inside the ZIP."
        )

    json_name = json_files[0]

    print(f"Found annotation file: {json_name}")

    with z.open(json_name) as f:
        data = json.load(f)


images = data["images"]
annotations = data["annotations"]
categories = data["categories"]


print("\nWCS dataset information:")
print(f"  Images      : {len(images):,}")
print(f"  Annotations : {len(annotations):,}")
print(f"  Categories  : {len(categories):,}")


# ============================================================
# 5. CATEGORY MAPPING
# ============================================================

category_id_to_name = {
    category["id"]: category["name"]
    for category in categories
}


name_to_category_id = {
    category["name"]: category["id"]
    for category in categories
}


print("\nChecking requested classes...")

for class_name in TARGET_CLASSES:

    if class_name not in name_to_category_id:
        print(f"WARNING: {class_name} was not found!")

    else:
        category_id = name_to_category_id[class_name]

        print(
            f"  FOUND: {class_name} "
            f"(category_id={category_id})"
        )


# ============================================================
# 6. MAP IMAGE IDs TO IMAGE INFORMATION
# ============================================================

image_id_to_image = {
    image["id"]: image
    for image in images
}


# ============================================================
# 7. GROUP ANNOTATIONS BY IMAGE
# ============================================================

print("\nOrganizing annotations...")

annotations_by_image = defaultdict(list)

for annotation in annotations:
    image_id = annotation["image_id"]
    annotations_by_image[image_id].append(annotation)


# ============================================================
# 8. FIND IMAGES FOR EACH TARGET CLASS
# ============================================================

print("\nFinding suitable images...")

class_images = defaultdict(list)

for image_id, image_annotations in annotations_by_image.items():

    image_info = image_id_to_image.get(image_id)

    if image_info is None:
        continue

    found_classes = set()

    for annotation in image_annotations:

        category_name = category_id_to_name.get(
            annotation["category_id"]
        )

        if category_name in TARGET_CLASSES:
            found_classes.add(category_name)

    for class_name in found_classes:
        class_images[class_name].append(image_info)


# ============================================================
# 9. SELECT DAY + NIGHT IMAGES
# ============================================================

random.seed(RANDOM_SEED)


def is_night(image):
    """
    Determines whether an image was captured at night.

    WCS metadata contains datetime information.
    If datetime is unavailable, the image is treated as unknown.
    """

    dt = image.get("datetime")

    if not dt:
        return None

    try:
        hour = int(dt[11:13])

        # Approximate camera-trap day/night split
        if 6 <= hour < 18:
            return False
        else:
            return True

    except Exception:
        return None


selected_images = {}
selected_image_ids = set()


for class_name, display_name in TARGET_CLASSES.items():

    candidates = class_images[class_name]

    print(
        f"\n{display_name.upper()} "
        f"({class_name})"
    )

    print(
        f"  Available images: {len(candidates):,}"
    )

    # Remove duplicate image IDs
    unique = {}

    for image in candidates:
        unique[image["id"]] = image

    candidates = list(unique.values())

    # Separate day/night
    day_images = []
    night_images = []
    unknown_images = []

    for image in candidates:

        night = is_night(image)

        if night is True:
            night_images.append(image)

        elif night is False:
            day_images.append(image)

        else:
            unknown_images.append(image)

    random.shuffle(day_images)
    random.shuffle(night_images)
    random.shuffle(unknown_images)

    # Try to create a balanced day/night set
    target_each = MAX_IMAGES_PER_CLASS // 2

    selected = []

    selected.extend(
        day_images[:target_each]
    )

    selected.extend(
        night_images[:target_each]
    )

    # Fill remaining slots if one side doesn't have enough
    if len(selected) < MAX_IMAGES_PER_CLASS:

        remaining_pool = (
            day_images[target_each:]
            + night_images[target_each:]
            + unknown_images
        )

        random.shuffle(remaining_pool)

        needed = (
            MAX_IMAGES_PER_CLASS
            - len(selected)
        )

        selected.extend(
            remaining_pool[:needed]
        )

    # Final safety limit
    selected = selected[:MAX_IMAGES_PER_CLASS]

    selected_images[class_name] = selected

    print(
        f"  Selected: {len(selected):,}"
    )

    print(
        f"  Day:      "
        f"{sum(is_night(x) is False for x in selected)}"
    )

    print(
        f"  Night:    "
        f"{sum(is_night(x) is True for x in selected)}"
    )

    # Add to global set
    for image in selected:
        selected_image_ids.add(image["id"])


# ============================================================
# 10. REMOVE DUPLICATE IMAGES
# ============================================================

final_images = []

seen_ids = set()

for class_name in TARGET_CLASSES:

    for image in selected_images[class_name]:

        image_id = image["id"]

        if image_id not in seen_ids:

            final_images.append(image)

            seen_ids.add(image_id)


print("\n" + "=" * 70)

print(
    f"Total unique images to download: "
    f"{len(final_images):,}"
)

print("=" * 70)


# ============================================================
# 11. BUILD FINAL COCO ANNOTATIONS
# ============================================================

print("\nPreparing annotations...")


# Only keep annotations belonging to our selected classes
target_category_ids = {
    name_to_category_id[class_name]
    for class_name in TARGET_CLASSES
}


final_annotations = []

for image in final_images:

    image_id = image["id"]

    for annotation in annotations_by_image.get(
        image_id,
        []
    ):

        category_id = annotation["category_id"]

        if category_id in target_category_ids:

            final_annotations.append(
                annotation.copy()
            )


# ============================================================
# 12. CREATE SIMPLIFIED CATEGORY LIST
# ============================================================

final_categories = []

for class_name, display_name in TARGET_CLASSES.items():

    category_id = name_to_category_id[class_name]

    final_categories.append(
        {
            "id": category_id,
            "name": display_name,
            "supercategory": "animal"
        }
    )


# ============================================================
# 13. SAVE COCO ANNOTATION FILE
# ============================================================

coco_output = {
    "info": {
        "description": "WildGuard Wildlife Detection Dataset",
        "source": "WCS Camera Traps via LILA BC"
    },

    "licenses": [],

    "images": final_images,

    "annotations": final_annotations,

    "categories": final_categories
}


annotation_output = (
    OUTPUT_DIR /
    "annotations_coco.json"
)


with open(
    annotation_output,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        coco_output,
        f,
        indent=2
    )


print(
    f"\nSaved annotations to:"
    f"\n  {annotation_output}"
)


# ============================================================
# 14. CREATE CLASS NAMES FILE
# ============================================================

classes_output = OUTPUT_DIR / "classes.txt"

with open(
    classes_output,
    "w",
    encoding="utf-8"
) as f:

    for display_name in TARGET_CLASSES.values():
        f.write(display_name + "\n")


print(
    f"Saved class list to:"
    f"\n  {classes_output}"
)


# ============================================================
# 15. DOWNLOAD FUNCTION
# ============================================================

session = requests.Session()


def download_image(image):

    image_id = image["id"]

    file_name = image["file_name"]

    # Construct official WCS cloud URL
    encoded_path = quote(
        file_name,
        safe="/"
    )

    url = BASE_URL + encoded_path

    # Preserve original directory structure
    output_path = IMAGE_DIR / file_name

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Don't download it again
    if output_path.exists():

        return (
            image_id,
            True,
            "already_exists"
        )

    try:

        response = session.get(
            url,
            timeout=60
        )

        if response.status_code != 200:

            return (
                image_id,
                False,
                f"HTTP {response.status_code}"
            )

        # Basic validation
        if len(response.content) < 1000:

            return (
                image_id,
                False,
                "file_too_small"
            )

        with open(
            output_path,
            "wb"
        ) as f:

            f.write(
                response.content
            )

        return (
            image_id,
            True,
            "downloaded"
        )

    except Exception as e:

        return (
            image_id,
            False,
            str(e)
        )


# ============================================================
# 16. DOWNLOAD ALL SELECTED IMAGES
# ============================================================

print("\n" + "=" * 70)
print("STARTING IMAGE DOWNLOAD")
print("=" * 70)

successful = 0
failed = []

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = [
        executor.submit(
            download_image,
            image
        )
        for image in final_images
    ]

    for future in tqdm(
        as_completed(futures),
        total=len(futures),
        desc="Downloading"
    ):

        image_id, success, status = future.result()

        if success:

            successful += 1

        else:

            failed.append(
                (image_id, status)
            )


# ============================================================
# 17. SAVE DOWNLOAD REPORT
# ============================================================

report = {
    "total_selected": len(final_images),
    "successful_downloads": successful,
    "failed_downloads": len(failed),
    "failed_images": [
        {
            "image_id": image_id,
            "error": error
        }
        for image_id, error in failed
    ]
}


report_output = OUTPUT_DIR / "download_report.json"

with open(
    report_output,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        report,
        f,
        indent=2
    )


# ============================================================
# 18. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

print(
    f"\nImages selected : {len(final_images):,}"
)

print(
    f"Images downloaded: {successful:,}"
)

print(
    f"Failed downloads : {len(failed):,}"
)

print(
    f"\nImages location:"
    f"\n  {IMAGE_DIR}"
)

print(
    f"\nAnnotations:"
    f"\n  {annotation_output}"
)

print(
    f"\nClass list:"
    f"\n  {classes_output}"
)

print(
    f"\nDownload report:"
    f"\n  {report_output}"
)


if failed:

    print("\nWARNING:")
    print(
        f"{len(failed)} images could not be downloaded."
    )

    print(
        "You can run the script again."
    )

    print(
        "Already downloaded images will be skipped."
    )

else:

    print(
        "\nALL SELECTED IMAGES DOWNLOADED SUCCESSFULLY!"
    )


print("\nWildGuard dataset retrieval finished.")
print("=" * 70)

input("\nPress Enter to exit...")