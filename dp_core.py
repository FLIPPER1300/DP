import json
from typing import Any, Dict, List, Sequence, Tuple, Optional
import numpy as np
import gymnasium as gym
from gymnasium import spaces

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]

NUM_CANDIDATES = 15
NUM_LINES_BELOW = 3

def load_json_data(json_path: str) -> Any:
    """Načíta JSON súbor s UTF-8 kódovaním."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_page_char_scale(annotations: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    """Vypočíta priemernú šírku a výšku znakov na stránke."""
    if not annotations:
        return 30.0, 30.0
    widths = [ann["bbox"][2] for ann in annotations]
    heights = [ann["bbox"][3] for ann in annotations]
    avg_w = float(np.mean(widths)) if widths else 30.0
    avg_h = float(np.mean(heights)) if heights else 30.0
    return (avg_w if avg_w > 1 else 30.0, avg_h if avg_h > 1 else 30.0)

def build_coco_index(
    coco_data: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    """Z coco anotácií spraví prehľadné indexy podľa image_id."""
    images_by_id: Dict[int, Dict[str, Any]] = {
        img["id"]: img for img in coco_data.get("images", [])
    }
    annots_by_image: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in images_by_id}
    for ann in coco_data.get("annotations", []):
        annots_by_image[ann["image_id"]].append(ann)
    return images_by_id, annots_by_image

def find_candidates(
    current_point: Point,
    remaining_points: Sequence[Point],
    page_char_scale: Tuple[float, float],
    num_candidates: int,
) -> List[Point]:
    if not remaining_points:
        return []
    points_arr = np.array(remaining_points, dtype=float)
    current_point_arr = np.array(current_point, dtype=float)
    avg_w, avg_h = page_char_scale
    candidates = []
    used_points = set()

    # PRIORITA 1: Body v aktuálnom riadku (vpravo od aktuálneho bodu)
    y_tolerance_same_line = avg_h * 0.8  # ZVYSENA TOLERANCIA PRE YOLO
    same_line_mask = (
        (points_arr[:, 0] > current_point_arr[0]) & 
        (np.abs(points_arr[:, 1] - current_point_arr[1]) < y_tolerance_same_line)
    )
    same_line_points = points_arr[same_line_mask]

    if len(same_line_points) > 0:
        indices = np.where(same_line_mask)[0]
        sorted_indices = indices[np.argsort(same_line_points[:, 0])]
        for idx in sorted_indices:
            pt = points_arr[idx]
            candidates.append(tuple(pt))
            used_points.add(tuple(pt))
            if len(candidates) >= num_candidates:
                return candidates

    for line_offset in range(1, NUM_LINES_BELOW + 1):
        line_y = current_point_arr[1] + line_offset * avg_h
        line_tolerance = avg_h * 0.8
        line_mask = np.abs(points_arr[:, 1] - line_y) < line_tolerance
        line_points = points_arr[line_mask]

        if len(line_points) > 0:
            indices = np.where(line_mask)[0]
            x_positions = line_points[:, 0]
            sorted_line_indices = indices[np.argsort(x_positions)]
            for idx in sorted_line_indices:
                pt = tuple(points_arr[idx])
                if pt not in used_points:
                    candidates.append(pt)
                    used_points.add(pt)
                    if len(candidates) >= num_candidates:
                        return candidates

    remaining_mask = np.ones(len(points_arr), dtype=bool)
    for used_pt in used_points:
        for i, pt in enumerate(points_arr):
            if abs(pt[0] - used_pt[0]) < 0.01 and abs(pt[1] - used_pt[1]) < 0.01:
                remaining_mask[i] = False
                break

    remaining_arr = points_arr[remaining_mask]
    if len(remaining_arr) > 0:
        distances = np.linalg.norm(remaining_arr - current_point_arr, axis=1)
        sorted_indices = np.argsort(distances)
        for idx in sorted_indices:
            pt = tuple(remaining_arr[idx])
            candidates.append(pt)
            if len(candidates) >= num_candidates:
                return candidates

    while len(candidates) < num_candidates:
        if candidates:
            candidates.append(candidates[-1])
        else:
            candidates.append(tuple(current_point_arr))
    return candidates[:num_candidates]


class BaseDataLoader:
    """Jednotný interface pre načítavanie a preprocessing dát (COCO aj YOLO)."""
    @staticmethod
    def parse_yolo(yolo_detections: List[Dict], confidence_threshold: float = 0.0) -> Tuple[List[Point], List[Dict]]:
        all_points = []
        all_annots = []
        used_points_dedup = set()
        
        # Filtrovanie nekvalitných detekcií
        filtered_dets = [d for d in yolo_detections if d.get("confidence", 1.0) >= confidence_threshold]
        if not filtered_dets:
            return [], []

        # Výpočet dynamickej výšky
        all_y = [d["centroid"][1] for d in filtered_dets]
        y_diffs = np.diff(sorted(all_y))
        row_diffs = [d for d in y_diffs if 20 < d < 200]
        computed_avg_h = float(np.mean(row_diffs)) if row_diffs else 80.0
        computed_avg_w = computed_avg_h * 0.8

        for y_ann in filtered_dets:
            cx, cy = float(y_ann["centroid"][0]), float(y_ann["centroid"][1])
            
            # NMS Fallback eliminácia absolútnych duplicít
            rounded_pt = (round(cx, 1), round(cy, 1))
            if rounded_pt in used_points_dedup:
                continue
            used_points_dedup.add(rounded_pt)

            bbox = y_ann.get("bbox", [])
            if len(bbox) == 4:
                w = float(bbox[2] - bbox[0])
                h = float(bbox[3] - bbox[1])
            else:
                w, h = computed_avg_w, computed_avg_h

            if w <= 0: w = computed_avg_w
            if h <= 0: h = computed_avg_h

            all_points.append((cx, cy))
            all_annots.append({
                "category_id": y_ann["class"],
                "yolo_label": str(y_ann.get("class", "")),
                "centroid": [cx, cy],
                "bbox": [cx - w/2, cy - h/2, w, h]
            })
        return all_points, all_annots

    @staticmethod
    def parse_coco(annotations: List[Dict], categories: Dict[int, str]) -> Tuple[List[Point], List[Dict]]:
        all_points = []
        all_annots = []
        for ann in annotations:
            bbox = tuple(ann["bbox"])
            label = categories.get(ann["category_id"], "")
            x, y, w, h = bbox
            cx = x + w / 2.0
            cy = y + h / 2.0
            # Offsety pre specific labels (z DP_main.py)
            if label == "1":
                cy += h * 0.1
            elif label == "6":
                cx -= w * 0.15
                cy += h * 0.15
            elif label == "9":
                cx += w * 0.15
                cy -= h * 0.2
            
            all_points.append((cx, cy))
            ann.update({"centroid": [cx, cy]})
            all_annots.append(ann)
        return all_points, all_annots


class ReadingEnv(gym.Env):
    metadata = {"render_modes": []}
    def __init__(self, num_candidates: int) -> None:
        super().__init__()
        self.num_candidates = num_candidates
        obs_dim = 2 * num_candidates
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(num_candidates)
        self.page_char_scale = (30.0, 30.0)
        self.all_page_points = []
        self.remaining_points = []
        self.current_point = None
        self.annotations = []

    def _empty_obs(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _build_observation(self) -> np.ndarray:
        if self.current_point is None:
            return self._empty_obs()
        candidates = find_candidates(self.current_point, self.remaining_points, self.page_char_scale, self.num_candidates)
        if not candidates:
            return self._empty_obs()
        current_point_arr = np.array(self.current_point, dtype=float)
        avg_w, avg_h = self.page_char_scale
        obs_vectors = []
        for cand in candidates:
            rel_vec = np.array(cand, dtype=float) - current_point_arr
            obs_vectors.extend([float(rel_vec[0] / avg_w), float(rel_vec[1] / avg_h)])
        return np.array(obs_vectors, dtype=np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if options is None or "all_points" not in options or "annotations" not in options:
            if hasattr(self, "expert_data_ref") and self.expert_data_ref:
                import random
                item = random.choice(self.expert_data_ref)
                img_file = item.get("image_file")
                img_id = getattr(self, "file_to_id_ref", {}).get(img_file)
                page_annotations = getattr(self, "coco_annots_ref", {}).get(img_id, [])
                all_points_on_page = [tuple(t["center"]) for t in item.get("trajectory", [])] if "trajectory" in item else [(0.0, 0.0)]
                if not all_points_on_page: all_points_on_page = [(0.0, 0.0)]
                options = {"all_points": all_points_on_page, "annotations": page_annotations}
            else:
                raise ValueError("Musíte poskytnúť 'all_points' a 'annotations' v options pri resete.")
        
        self.all_page_points = list(map(tuple, options["all_points"]))
        self.annotations = list(options["annotations"])
        self.page_char_scale = compute_page_char_scale(self.annotations)
        if not self.all_page_points:
            self.current_point = (0.0, 0.0)
            self.remaining_points = []
            return self._empty_obs(), {}
        
        points_arr = np.array(self.all_page_points, dtype=float)
        top_idx = int(np.argmin(points_arr[:, 1]))
        top_y = float(points_arr[top_idx, 1])
        _, avg_h = self.page_char_scale
        first_line_candidates = [p for p in self.all_page_points if abs(p[1] - top_y) < avg_h * 0.75]
        start_point = min(first_line_candidates, key=lambda p: p[0])
        self.current_point = (float(start_point[0]), float(start_point[1]))
        self.remaining_points = [p for p in self.all_page_points if p != self.current_point]
        return self._build_observation(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        if self.current_point is None:
            return self._empty_obs(), 0.0, True, False, {}
        candidates = find_candidates(self.current_point, self.remaining_points, self.page_char_scale, self.num_candidates)
        if not candidates or action >= len(candidates):
            return self._build_observation(), 0.0, True, False, {}
        chosen_point = candidates[action]
        self.current_point = chosen_point
        self.remaining_points = [p for p in self.remaining_points if tuple(p) != tuple(chosen_point)]
        done = len(self.remaining_points) == 0
        return self._build_observation(), 0.0, done, False, {}

