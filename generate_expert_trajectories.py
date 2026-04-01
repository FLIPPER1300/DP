import json
import os
import cv2
import numpy as np

def compute_centroid(bbox):
    x, y, w, h = bbox
    return x + w / 2, y + h / 2

def draw_trajectories_from_json(expert_json_path, image_dir, draw_dir="image_trajectories_from_experts"):
    # Načítanie expert trajektórií
    with open(expert_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    os.makedirs(draw_dir, exist_ok=True)

    for entry in data:
        file_name = entry["image_file"]
        image_path = os.path.join(image_dir, file_name)
        traj = entry.get("trajectory", [])
        expert_pairs = entry.get("expert_pairs", [])

        if not os.path.exists(image_path):
            print(f"⚠️ Obrázok {image_path} sa nenašiel, preskakujem vizualizáciu.")
            continue

        img = cv2.imread(image_path)

        # --- Nakresli body a ich labely ---
        for t in traj:
            cx, cy = map(int, t["center"])
            label = t["label"]
            cv2.circle(img, (cx, cy), 10, (0, 0, 255), -1)
            cv2.putText(img, str(label), (cx + 15, cy - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # --- Nakresli čiary podľa expert_pairs ---
        for pair in expert_pairs:
            sx, sy = map(int, pair["state"])
            ax, ay = map(int, pair["action"])
            cv2.line(img, (sx, sy), (ax, ay), (0, 255, 0), 3)

        # --- Uloženie ---
        out_path = os.path.join(draw_dir, file_name)
        cv2.imwrite(out_path, img)

    print(f"✅ Vizualizácie (z expert_trajectories.json) uložené do: {draw_dir}")


def refine_rows_with_linefit(traj, row_threshold, iters=3):
    """
    vstup:
      traj - list položiek s "center": [cx, cy] a "bbox"
      row_threshold - prah na inicialne priradenie (typicky niečo ako avg_height * factor)
      iters - počet iterácií pre reassignment
    výstup:
      rows - zoznam zoznamov (každý row je list položiek)
    """
    # --- počiatočná grupácia podľa y (podobná tvojej pôvodnej) ---
    traj_sorted = sorted(traj, key=lambda t: t["center"][1])
    rows = []
    for t in traj_sorted:
        placed = False
        for row in rows:
            # porovnávať so stredom prvého prvku riadku (ako doteraz)
            if abs(t["center"][1] - row[0]["center"][1]) <= row_threshold:
                row.append(t)
                placed = True
                break
        if not placed:
            rows.append([t])

    # --- iteratívne vylepšovanie pomocou lineárneho fitu ---
    for _ in range(iters):
        # pre každý row spočítať lineárny fit (a,b) y = a*x + b
        fits = []
        for row in rows:
            xs = np.array([p["center"][0] for p in row], dtype=float)
            ys = np.array([p["center"][1] for p in row], dtype=float)
            if len(xs) >= 2:
                a, b = np.polyfit(xs, ys, 1)  # lineárny fit
            else:
                # pre single point: horizontálna čiaru na jeho y
                a, b = 0.0, float(ys.mean())
            fits.append((a, b))

        # pripraviť nové empty rows ako samostatné lists (budú naplnené podľa najlepšieho fitu)
        new_rows = [[] for _ in rows]

        # pre každý bod nájsť fit s minimalnym zvislým rozdielom
        for p in traj:
            cx, cy = float(p["center"][0]), float(p["center"][1])
            best_idx = None
            best_dist = None
            for idx, (a, b) in enumerate(fits):
                y_pred = a * cx + b
                dist = abs(cy - y_pred)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_idx = idx
            # ak najlepší fit je v rámci určitej vzdialenosti, priraď; inak vytvor nový row
            # tu použijeme adaptívny prah = row_threshold (môžeš ho zmeniť)
            if best_dist is not None and best_dist <= row_threshold * 0.8:
                new_rows[best_idx].append(p)
            else:
                # nález nového samostatného riadku (každý bod ktorý nesedí do žiadneho fitu)
                new_rows.append([p])

        # odstrániť prázdne rows a nahradiť rows
        rows = [r for r in new_rows if len(r) > 0]

    # --- finálne zoradenie: podľa mediánu (alebo priemeru) predikovaného y, a v riadku podľa x ---
    final_rows = []
    for row in rows:
        row.sort(key=lambda t: t["center"][0])  # zoradiť podľa x v riadku
        final_rows.append(row)

    # zoradiť riadky podĺa mediánu y (alebo podľa fitu)
    final_rows.sort(key=lambda r: np.median([p["center"][1] for p in r]))
    return final_rows

def generate_expert_trajectories(json_path, image_dir, output_path=None, draw_dir="image_trajectories"):
    # Načítanie COCO-like JSON
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = {img["id"]: img for img in data["images"]}
    annotations = data["annotations"]

    # Nové: mapovanie category_id -> name
    category_map = {c["id"]: c["name"] for c in data.get("categories", [])}

    trajectories = []
    os.makedirs(draw_dir, exist_ok=True)

    for img_id, img_info in images.items():
        file_name = img_info["file_name"]
        image_path = os.path.join(image_dir, file_name)

        annots = [a for a in annotations if a["image_id"] == img_id]
        if not annots:
            continue

        traj = []
        for i, a in enumerate(annots):
            bbox = a["bbox"]
            category_id = a["category_id"]

            # Použitie mena kategórie
            label = category_map.get(category_id, str(category_id))

            cx, cy = compute_centroid(bbox)

            if label == "9":
                cx += bbox[2] * 0.15
                cy -= bbox[3] * 0.2
            if label == "6":
                cx -= bbox[2] * 0.15
                cy += bbox[3] * 0.15
            if label == "1":
                cy += bbox[3] * 0.1

            traj.append({
                "id": i,
                "center": [float(cx), float(cy)],
                "label": label,
                "bbox": bbox
            })

        avg_height = sum([b["bbox"][3] for b in annots]) / len(annots) * 1.2
        row_threshold = avg_height

        rows = refine_rows_with_linefit(traj, row_threshold, iters=3)
        sorted_traj = []
        for row in rows:
            row.sort(key=lambda t: t["center"][0])
            sorted_traj.extend(row)
        traj = sorted_traj

        # traj.sort(key=lambda t: t["center"][1])
        # rows = []
        # for t in traj:
        #     placed = False
        #     for row in rows:
        #         if abs(t["center"][1] - row[0]["center"][1]) <= row_threshold:
        #             row.append(t)
        #             placed = True
        #             break
        #     if not placed:
        #         rows.append([t])
        #
        # sorted_traj = []
        # for row in rows:
        #     row.sort(key=lambda t: t["center"][0])
        #     sorted_traj.extend(row)
        # traj = sorted_traj

        expert_pairs = []
        for i in range(len(traj) - 1):
            expert_pairs.append({
                "state": traj[i]["center"],
                "action": traj[i + 1]["center"]
            })

        trajectories.append({
            "image_file": file_name,
            "trajectory": traj,
            "expert_pairs": expert_pairs
        })

    return trajectories


if __name__ == "__main__":
    input_json = "_annotations.coco.json"
    image_dir = "images"
    output_json = "expert_trajectories.json"

    # generate_expert_trajectories(input_json, image_dir, output_json)

    draw_trajectories_from_json(
        expert_json_path="expert_trajectories_repaired.json",
        image_dir="images",
        draw_dir="image_trajectories_from_expert"
    )