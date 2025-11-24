import csv
import gzip
import json
from pathlib import Path

CATALOG_DIR = Path("catalog")
EXISTING_CATALOG = CATALOG_DIR / "catalog.csv"
# MERGED_CATALOG = CATALOG_DIR / "catalog_merged.csv"
MERGED_CATALOG = EXISTING_CATALOG

ABO_IMAGES_META = CATALOG_DIR / "abo-images-small" / "images" / "metadata" / "images.csv.gz"
ABO_IMAGES_ROOT = CATALOG_DIR / "abo-images-small" / "images" / "small"
ABO_LISTINGS_META_DIR = CATALOG_DIR / "abo-listings" / "listings" / "metadata"

IMAGES_TARGET_DIR = CATALOG_DIR / "images"
IMAGES_TARGET_DIR.mkdir(parents=True, exist_ok=True)


def choose_text_field(field_list, default=""):
    """Pick English value if available, else first, else default."""
    if not field_list:
        return default
    # Try English
    for item in field_list:
        lang = item.get("language_tag", "") or ""
        if lang.lower().startswith("en"):
            return item.get("value", default)
    # Fallback: first value
    return field_list[0].get("value", default)


def load_image_meta():
    """Return dict: image_id -> path"""
    mapping = {}
    with gzip.open(ABO_IMAGES_META, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_id = row["image_id"]
            path = row["path"]
            mapping[image_id] = path
    return mapping


def iter_listings():
    """Yield product dicts from all listings_*.json.gz files."""
    for gz_path in sorted(ABO_LISTINGS_META_DIR.glob("listings_*.json.gz")):
        with gzip.open(gz_path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)


def main():
    # 1) Load existing rows
    rows = []
    with open(EXISTING_CATALOG, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # 2) Image metadata
    image_meta = load_image_meta()

    # 3) Process ABO listings
    for prod in iter_listings():
        item_id = prod.get("item_id")
        domain_name = prod.get("domain_name")
        main_image_id = prod.get("main_image_id")

        if not item_id or not domain_name or not main_image_id:
            continue

        # Map image id -> path
        rel_path = image_meta.get(main_image_id)
        if not rel_path:
            continue

        # Resolve actual source image file
        src_img = ABO_IMAGES_ROOT / rel_path
        if not src_img.is_file():
            continue

        # Build sku_id
        safe_domain = domain_name.replace(".", "_")  # "amazon.com" -> "amazon_com"
        sku_id = f"abo_{safe_domain}_{item_id}"  # abo_amazon_com_B08332HDLR

        # Title & brand
        title = choose_text_field(prod.get("item_name", []), default=item_id)
        brand = choose_text_field(prod.get("brand", []), default="Unknown")

        # Copy image into catalog/images as <sku_id>.jpg
        dst_img = IMAGES_TARGET_DIR / f"{sku_id}.jpg"

        # If you don't want to overwrite existing, you can guard with:
        # if dst_img.exists(): continue

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_img.write_bytes(src_img.read_bytes())

        image_path = f"images/{dst_img.name}"

        rows.append({
            "sku_id": sku_id,
            "title": title,
            "brand": brand,
            "image_path": image_path,
        })

    # 4) Write merged CSV
    fieldnames = ["sku_id", "title", "brand", "image_path"]
    with open(MERGED_CATALOG, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
