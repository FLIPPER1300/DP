import json
import os

def compute_centroid(bbox):
    x, y, w, h = bbox
    return [float(x + w / 2), float(y + h / 2)]

def save_image_jsons(coco_json_path, output_dir="image_json_data"):
    # Načítanie COCO-like JSON
    with open(coco_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    annotations = data["annotations"]

    # Dôležité: mapovanie category_id → name
    category_map = {c["id"]: c["name"] for c in data.get("categories", [])}

    os.makedirs(output_dir, exist_ok=True)

    for img_id, img_info in images.items():
        file_name = img_info["file_name"]

        annots = [a for a in annotations if a["image_id"] == img_id]

        img_json = []
        for i, a in enumerate(annots):
            bbox = a["bbox"]
            center = compute_centroid(bbox)

            # Použitie mena kategórie
            label = category_map.get(a["category_id"], str(a["category_id"]))

            img_json.append({
                "id": i,
                "center": center,
                "label": label,
                "bbox": bbox
            })

        base_name = os.path.splitext(file_name)[0]
        out_path = os.path.join(output_dir, f"{base_name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(img_json, f, indent=2, ensure_ascii=False)

    print(f"✅ Všetky JSONy uložené do priečinka: {output_dir}")


if __name__ == "__main__":
    coco_json_path = "_annotations.coco.json"
    save_image_jsons(coco_json_path)
