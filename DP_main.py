import json
import os
from typing import Any, Dict, List, Sequence, Tuple, Optional
import difflib
import cv2
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from imitation.algorithms import bc
from imitation.data import types as im_types
from stable_baselines3.common.logger import configure
from stable_baselines3.common.policies import ActorCriticPolicy

try:
    from rapidfuzz import fuzz, distance
    from rapidfuzz.distance import Levenshtein
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not available. Install with: pip install rapidfuzz")

# --- Konfigurácia GPU ---
try:
    import torch_directml
    DEVICE = torch_directml.device()
    print("[GPU] AMD GPU (DirectML) dostupné")
except ImportError:
    print("[GPU] torch_directml nie je nainštalovaný, skúšam CUDA...")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
        print("[GPU] CUDA dostupné")
    else:
        DEVICE = torch.device("cpu")
        print("[GPU] Používam CPU (GPU nie je dostupné)")

# ---------------- Konfigurácia ----------------

EXPERT_JSON = "expert_trajectories_repaired.json"
COCO_JSON = "_annotations.coco.json"
IMAGE_DIR = "images"
OUTDIR = "image_bc_traj"
BC_MODEL_PATH = "bc_model.pt"
LOG_DIR = "logs/bc_reader_final"

TRAIN_BC = True
TRAIN_DAGGER = True
TRAIN_GAIL = True
GAIL_START_MODEL = "dagger"  # prepínač: "bc" alebo "dagger". Určuje, aký model sa použije na začiatku GAIL
N_EPOCHS = 20
NUM_CANDIDATES = 15
NUM_LINES_BELOW = 3  # Počet riadkov pod aktuálnym, v ktorých hľadáme kandidátov
DAGGER_ITERATIONS = 2  # Počet iterácií

# Debug nastavenia
DEBUG_CANDIDATES = False  # Ak True, vypíše info o kandidátoch pri generovaní
DEBUG_TRAJECTORY = False  # Ak True, vypíše detaily o každom kroku trajektórieí

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]


# ---------------- Pomocné funkcie ----------------

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

    return (avg_w if avg_w > 1 else 30.0,
            avg_h if avg_h > 1 else 30.0)


def build_coco_index(
    coco_data: Dict[str, Any],
) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    """Z coco anotácií spraví prehľadné indexy podľa image_id."""
    images_by_id: Dict[int, Dict[str, Any]] = {
        img["id"]: img for img in coco_data["images"]
    }

    annots_by_image: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in images_by_id}
    for ann in coco_data["annotations"]:
        annots_by_image[ann["image_id"]].append(ann)

    return images_by_id, annots_by_image


def find_candidates(
    current_point: Point,
    remaining_points: Sequence[Point],
    page_char_scale: Tuple[float, float],
    num_candidates: int,
) -> List[Point]:
    """
    Inteligentný výber K kandidátov s prioritami:
    1. Body v aktuálnom riadku (vpravo) - pokračovanie čítania
    2. Body na začiatku ďalších riadkov - skákanie medzi riadkami
    3. Najbližší body globálne - fallback

    Toto zaistí logické čítanie s možnosťou skákať na ďalší riadok.
    """
    if not remaining_points:
        return []

    points_arr = np.array(remaining_points, dtype=float)
    current_point_arr = np.array(current_point, dtype=float)
    avg_w, avg_h = page_char_scale

    candidates = []
    used_points = set()

    # PRIORITA 1: Body v aktuálnom riadku (vpravo od aktuálneho bodu)
    # Tolerancia: ±40% výšky znaku
    y_tolerance_same_line = avg_h * 0.4
    same_line_mask = (
        (points_arr[:, 0] > current_point_arr[0]) &  # Vpravo
        (np.abs(points_arr[:, 1] - current_point_arr[1]) < y_tolerance_same_line)  # Podobná výška
    )
    same_line_points = points_arr[same_line_mask]

    if len(same_line_points) > 0:
        # Zoraď podľa x-pozície (ľava-doprava)
        indices = np.where(same_line_mask)[0]
        sorted_indices = indices[np.argsort(same_line_points[:, 0])]
        for idx in sorted_indices:
            pt = points_arr[idx]
            candidates.append(tuple(pt))
            used_points.add(tuple(pt))
            if len(candidates) >= num_candidates:
                return candidates

    # PRIORITA 2: Body na začiatku ďalších riadkov (skákanie medzi riadkami)
    # Hľadaj v NUM_LINES_BELOW riadkoch nižšie
    for line_offset in range(1, NUM_LINES_BELOW + 1):
        # Presná pozícia riadka
        line_y = current_point_arr[1] + line_offset * avg_h

        # Tolerancia: ±50% výšky znaku
        line_tolerance = avg_h * 0.8
        line_mask = np.abs(points_arr[:, 1] - line_y) < line_tolerance
        line_points = points_arr[line_mask]

        if len(line_points) > 0:
            # Na tomto riadku: zoraď podľa x-pozície od začiatku
            indices = np.where(line_mask)[0]
            x_positions = line_points[:, 0]
            sorted_line_indices = indices[np.argsort(x_positions)]

            for idx in sorted_line_indices:
                pt = points_arr[idx]
                pt_tuple = tuple(pt)
                if pt_tuple not in used_points:
                    candidates.append(pt_tuple)
                    used_points.add(pt_tuple)
                    if len(candidates) >= num_candidates:
                        return candidates

    # PRIORITA 3: Globálne najbližší body (fallback)
    # Vypočítaj vzdialenosti ku všetkým zvyšným bodom
    remaining_mask = np.ones(len(points_arr), dtype=bool)
    for used_pt in used_points:
        # Nájdi index bodu v points_arr
        for i, pt in enumerate(points_arr):
            if abs(pt[0] - used_pt[0]) < 0.01 and abs(pt[1] - used_pt[1]) < 0.01:
                remaining_mask[i] = False
                break

    remaining_arr = points_arr[remaining_mask]
    if len(remaining_arr) > 0:
        distances = np.linalg.norm(remaining_arr - current_point_arr, axis=1)
        sorted_indices = np.argsort(distances)

        for idx in sorted_indices:
            pt = remaining_arr[idx]
            pt_tuple = tuple(pt)
            candidates.append(pt_tuple)
            if len(candidates) >= num_candidates:
                return candidates

    # Vráť presne num_candidates (doplň posledným ak je menej)
    while len(candidates) < num_candidates:
        if candidates:
            candidates.append(candidates[-1])
        else:
            candidates.append(tuple(current_point_arr))

    return candidates[:num_candidates]




# ---------------- Prostredie ----------------

class ReadingEnv(gym.Env):
    """
    Prostredie, kde agent číta body na stránke.

    Pozorovanie:
      - vektor relatívnych pozícií kandidátov k aktuálnemu bodu,
        škálovaný podľa priemernej veľkosti znakov (x/avg_w, y/avg_h)

    Akcie:
      - index zvoleného kandidáta (Discrete(num_candidates))
    """

    metadata = {"render_modes": []}

    def __init__(self, num_candidates: int) -> None:
        super().__init__()

        self.num_candidates = num_candidates
        obs_dim = 2 * num_candidates  # (dx, dy) pre každý kandidát

        self.observation_space = spaces.Box(
            low=-100.0,
            high=100.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(num_candidates)

        # (avg_w, avg_h)
        self.page_char_scale: Tuple[float, float] = (30.0, 30.0)

        self.all_page_points: List[Point] = []
        self.remaining_points: List[Point] = []
        self.current_point: Optional[Point] = None
        self.annotations: List[Dict[str, Any]] = []

    # --- vnútorné pomocné metódy ---

    def _empty_obs(self) -> np.ndarray:
        return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _build_observation(self) -> np.ndarray:
        """Vytvorí pozorovanie z aktuálneho bodu a kandidátov."""
        if self.current_point is None:
            return self._empty_obs()

        candidates = find_candidates(
            current_point=self.current_point,
            remaining_points=self.remaining_points,
            page_char_scale=self.page_char_scale,
            num_candidates=self.num_candidates,
        )

        if not candidates:
            return self._empty_obs()

        current_point_arr = np.array(self.current_point, dtype=float)
        avg_w, avg_h = self.page_char_scale

        obs_vectors: List[float] = []
        for cand in candidates:
            cand_arr = np.array(cand, dtype=float)
            rel_vec = cand_arr - current_point_arr
            # škálovanie podľa priemernej veľkosti znaku
            obs_vectors.extend([
                float(rel_vec[0] / avg_w),
                float(rel_vec[1] / avg_h),
            ])

        return np.array(obs_vectors, dtype=np.float32)

    # --- Gym API ---

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        if options is None or "all_points" not in options or "annotations" not in options:
            if hasattr(self, "expert_data_ref") and self.expert_data_ref:
                import random
                item = random.choice(self.expert_data_ref)
                img_file = item.get("image_file")
                img_id = getattr(self, "file_to_id_ref", {}).get(img_file)
                page_annotations = getattr(self, "coco_annots_ref", {}).get(img_id, [])
                all_points_on_page = [tuple(t["center"]) for t in item["trajectory"]] if "trajectory" in item else [(0.0, 0.0)]
                if not all_points_on_page:
                    all_points_on_page = [(0.0, 0.0)]
                options = {"all_points": all_points_on_page, "annotations": page_annotations}
            else:
                raise ValueError(
                    "Musíte poskytnúť 'all_points' a 'annotations' v options pri resete."
                )

        self.all_page_points = list(map(tuple, options["all_points"]))
        self.annotations = list(options["annotations"])

        # aktualizuj škálu znakov podľa anotácií
        self.page_char_scale = compute_page_char_scale(self.annotations)

        if not self.all_page_points:
            self.current_point = (0.0, 0.0)
            self.remaining_points = []
            return self._empty_obs(), {}

        # nájdi najvyšší bod (najmenšie y)
        points_arr = np.array(self.all_page_points, dtype=float)
        top_idx = int(np.argmin(points_arr[:, 1]))
        top_y = float(points_arr[top_idx, 1])

        # prvý riadok: body, ktoré majú y blízko top_y
        _, avg_h = self.page_char_scale
        first_line_candidates = [
            p for p in self.all_page_points
            if abs(p[1] - top_y) < avg_h * 0.75
        ]

        # štartovací bod: najviac vľavo
        start_point = min(first_line_candidates, key=lambda p: p[0])

        self.current_point = (float(start_point[0]), float(start_point[1]))
        self.remaining_points = [
            p for p in self.all_page_points if p != self.current_point
        ]

        obs = self._build_observation()
        info: Dict[str, Any] = {}
        return obs, info

    def step(
        self,
        action: int,
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Posunie sa na bod zvolený akciou medzi kandidátmi."""
        if self.current_point is None:
            return self._empty_obs(), 0.0, True, False, {}

        candidates = find_candidates(
            current_point=self.current_point,
            remaining_points=self.remaining_points,
            page_char_scale=self.page_char_scale,
            num_candidates=self.num_candidates,
        )

        if not candidates or action >= len(candidates):
            # neplatná akcia alebo žiadni kandidáti -> koniec epizódy
            return self._build_observation(), 0.0, True, False, {}

        chosen_point = candidates[action]

        self.current_point = chosen_point
        self.remaining_points = [
            p for p in self.remaining_points if tuple(p) != tuple(chosen_point)
        ]

        done = len(self.remaining_points) == 0
        obs = self._build_observation()
        reward = 0.0
        truncated = False
        info: Dict[str, Any] = {}

        return obs, reward, done, truncated, info


# ---------------- Príprava dát + Tréning BC ----------------

def build_observation_from_candidates(
    current_point: Point,
    candidates: Sequence[Point],
    page_char_scale: Tuple[float, float],
) -> np.ndarray:
    """Spraví pozorovanie (vektor) z aktuálneho bodu a kandidátov."""
    current_point_arr = np.array(current_point, dtype=float)
    avg_w, avg_h = page_char_scale

    obs_vectors: List[float] = []
    for cand in candidates:
        cand_arr = np.array(cand, dtype=float)
        rel_vec = cand_arr - current_point_arr
        obs_vectors.extend([
            float(rel_vec[0] / avg_w),
            float(rel_vec[1] / avg_h),
        ])

    return np.array(obs_vectors, dtype=np.float32)


def diagnose_candidate_coverage(
    expert_data: List[Dict[str, Any]],
    coco_data: Dict[str, Any],
    num_candidates: int,
) -> Dict[str, Any]:
    """
    Diagnostika: Overí pokrytie expert_pair kandidátmi.

    DÔLEŽITÉ: expert_pairs sú ručne upravené a neusia byť v poradí trajectory!
    Táto diagnostika skontroluje, či expert_next_point je dostupný v candidates.
    """
    total_pairs = 0
    covered_pairs = 0
    uncovered_samples = []
    distances_to_expert = []

    coco_images, coco_annots = build_coco_index(coco_data)
    file_to_id: Dict[str, int] = {
        img_info["file_name"]: img_id
        for img_id, img_info in coco_images.items()
    }

    for item in expert_data:
        img_file = item.get("image_file")
        if not img_file:
            continue

        img_id = file_to_id.get(img_file)
        if img_id is None:
            continue

        page_annotations = coco_annots.get(img_id, [])
        if not page_annotations:
            continue

        page_char_scale = compute_page_char_scale(page_annotations)
        all_points_on_page: List[Point] = [
            tuple(t["center"]) for t in item["trajectory"]
        ]

        for pair_idx, pair in enumerate(item["expert_pairs"]):
            total_pairs += 1
            current_point: Point = tuple(pair["state"])
            expert_next_point: Point = tuple(pair["action"])

            # Overíme či expert_next_point je vôbec na stránke
            if expert_next_point not in all_points_on_page:
                uncovered_samples.append({
                    "reason": "expert_point_not_on_page",
                    "img_file": img_file,
                    "pair_idx": pair_idx,
                    "expert_point": expert_next_point,
                    "current_point": current_point,
                })
                continue

            remaining_points = [
                p for p in all_points_on_page if tuple(p) != tuple(current_point)
            ]

            candidates = find_candidates(
                current_point=current_point,
                remaining_points=remaining_points,
                page_char_scale=page_char_scale,
                num_candidates=num_candidates,
            )

            if not candidates:
                uncovered_samples.append({
                    "reason": "no_candidates",
                    "img_file": img_file,
                    "pair_idx": pair_idx,
                    "expert_point": expert_next_point,
                })
                continue

            try:
                action_idx = candidates.index(expert_next_point)
                covered_pairs += 1
                distances_to_expert.append(0.0)
            except ValueError:
                # Expert bod nie je v kandidátoch
                closest_candidate = min(
                    candidates,
                    key=lambda c: np.linalg.norm(
                        np.array(c) - np.array(expert_next_point)
                    )
                )
                dist = float(np.linalg.norm(
                    np.array(closest_candidate) - np.array(expert_next_point)
                ))
                distances_to_expert.append(dist)

                uncovered_samples.append({
                    "reason": "expert_not_in_candidates",
                    "img_file": img_file,
                    "pair_idx": pair_idx,
                    "expert_point": expert_next_point,
                    "current_point": current_point,
                    "num_candidates": len(candidates),
                    "closest_candidate": closest_candidate,
                    "distance_to_expert": dist,
                    "page_char_scale": page_char_scale,
                })

    coverage_percent = (covered_pairs / total_pairs * 100) if total_pairs > 0 else 0.0
    avg_distance = float(np.mean(distances_to_expert)) if distances_to_expert else 0.0
    max_distance = float(np.max(distances_to_expert)) if distances_to_expert else 0.0

    return {
        "total_pairs": total_pairs,
        "covered_pairs": covered_pairs,
        "coverage_percent": coverage_percent,
        "uncovered_pairs": len(uncovered_samples),
        "avg_distance_to_expert": avg_distance,
        "max_distance_to_expert": max_distance,
        "uncovered_samples": uncovered_samples[:15],  # prvých 15
    }


def prepare_bc_data(
    expert_data: List[Dict[str, Any]],
    coco_data: Dict[str, Any],
    num_candidates: int,
) -> im_types.Transitions:
    """Pripraví Transitions objekt pre behavior cloning z expert dát."""
    observations: List[np.ndarray] = []
    actions: List[int] = []

    coco_images, coco_annots = build_coco_index(coco_data)

    # mapovanie file_name -> image_id
    file_to_id: Dict[str, int] = {
        img_info["file_name"]: img_id
        for img_id, img_info in coco_images.items()
    }

    for item in expert_data:
        img_file = item.get("image_file")
        if not img_file:
            continue

        img_id = file_to_id.get(img_file)
        if img_id is None:
            continue

        page_annotations = coco_annots.get(img_id, [])
        if not page_annotations:
            continue

        page_char_scale = compute_page_char_scale(page_annotations)

        all_points_on_page: List[Point] = [
            tuple(t["center"]) for t in item["trajectory"]
        ]

        for pair in item["expert_pairs"]:
            current_point: Point = tuple(pair["state"])
            expert_next_point: Point = tuple(pair["action"])

            remaining_points = [
                p for p in all_points_on_page if tuple(p) != tuple(current_point)
            ]

            candidates = find_candidates(
                current_point=current_point,
                remaining_points=remaining_points,
                page_char_scale=page_char_scale,
                num_candidates=num_candidates,
            )
            if not candidates:
                continue

            # expert akcia musí byť medzi kandidátmi
            try:
                action_index = candidates.index(expert_next_point)
            except ValueError:
                # expert zvolil bod mimo našich kandidátov
                continue

            obs = build_observation_from_candidates(
                current_point=current_point,
                candidates=candidates,
                page_char_scale=page_char_scale,
            )

            observations.append(obs)
            actions.append(action_index)

    observations_arr = np.array(observations, dtype=np.float32)
    actions_arr = np.array(actions, dtype=np.int64)

    infos_arr = np.array([{} for _ in actions_arr])
    dones_arr = np.ones(len(actions_arr), dtype=bool)

    # next_obs = obs, lebo ide o one-step predikciu
    return im_types.Transitions(
        obs=observations_arr,
        acts=actions_arr,
        infos=infos_arr,
        dones=dones_arr,
        next_obs=observations_arr,
    )


# ============ FUNKCIe PRE DAGGER ============

def evaluate_model_accuracy(
    expert_data: List[Dict[str, Any]],
    coco_data: Dict[str, Any],
    model: ActorCriticPolicy,
    num_candidates: int,
) -> float:
    """Vyhodnotí one-step presnosť modelu oproti expertovi."""
    correct_predictions = 0
    total_predictions = 0

    coco_images, coco_annots = build_coco_index(coco_data)

    file_to_id: Dict[str, int] = {
        img_info["file_name"]: img_id
        for img_id, img_info in coco_images.items()
    }

    for item in expert_data:
        img_file = item.get("image_file")
        if not img_file:
            continue

        img_id = file_to_id.get(img_file)
        if img_id is None:
            continue

        page_annotations = coco_annots.get(img_id, [])
        if not page_annotations:
            continue

        page_char_scale = compute_page_char_scale(page_annotations)

        all_points_on_page: List[Point] = [
            tuple(t["center"]) for t in item["trajectory"]
        ]

        for pair in item["expert_pairs"]:
            current_point: Point = tuple(pair["state"])
            expert_next_point: Point = tuple(pair["action"])

            total_predictions += 1

            remaining_points = [
                p for p in all_points_on_page if tuple(p) != tuple(current_point)
            ]

            candidates = find_candidates(
                current_point=current_point,
                remaining_points=remaining_points,
                page_char_scale=page_char_scale,
                num_candidates=num_candidates,
            )
            if not candidates:
                continue

            try:
                expert_action_index = candidates.index(expert_next_point)
            except ValueError:
                total_predictions -= 1
                continue

            obs = build_observation_from_candidates(
                current_point=current_point,
                candidates=candidates,
                page_char_scale=page_char_scale,
            )

            model_action_index, _ = model.predict(obs, deterministic=True)

            if int(model_action_index) == int(expert_action_index):
                correct_predictions += 1

    if total_predictions == 0:
        return 0.0

    return 100.0 * correct_predictions / total_predictions


def dagger_iteration(
    expert_data: List[Dict[str, Any]],
    coco_data: Dict[str, Any],
    model: ActorCriticPolicy,
    num_candidates: int,
    iteration: int,
) -> Optional[im_types.Transitions]:
    """DAgger iterácia - model generuje, expert opravuje."""
    print(f"\n{'='*70}")
    print(f"DAgger ITERÁCIA {iteration}")
    print(f"{'='*70}")

    observations: List[np.ndarray] = []
    actions: List[int] = []

    corrections_found = 0
    total_steps = 0

    coco_images, coco_annots = build_coco_index(coco_data)
    file_to_id: Dict[str, int] = {
        img_info["file_name"]: img_id
        for img_id, img_info in coco_images.items()
    }

    for item_idx, item in enumerate(expert_data):
        img_file = item.get("image_file")
        if not img_file:
            continue

        img_id = file_to_id.get(img_file)
        if img_id is None:
            continue

        page_annotations = coco_annots.get(img_id, [])
        if not page_annotations:
            continue

        page_char_scale = compute_page_char_scale(page_annotations)
        all_points_on_page: List[Point] = [
            tuple(t["center"]) for t in item["trajectory"]
        ]

        # Expert trajektóriu
        expert_trajectory = []
        if item.get("expert_pairs"):
            visited = set()
            current = tuple(item["expert_pairs"][0]["state"])
            expert_trajectory.append(current)
            visited.add(current)

            for pair in item["expert_pairs"]:
                next_point = tuple(pair["action"])
                if next_point not in visited:
                    expert_trajectory.append(next_point)
                    visited.add(next_point)
        else:
            expert_trajectory = all_points_on_page.copy()

        # Model trajektóriu
        env.reset(
            options={
                "all_points": all_points_on_page,
                "annotations": page_annotations,
            }
        )

        model_traj: List[Point] = []
        if env.current_point is not None:
            model_traj.append(env.current_point)

        obs = env._build_observation()
        done = False
        step = 0
        max_steps = len(all_points_on_page)

        while not done and step < max_steps:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(int(action))

            if env.current_point is not None:
                model_traj.append(env.current_point)
                step += 1

        # Porovnanie
        visited_model = set(model_traj)
        for expert_point in expert_trajectory:
            if expert_point not in visited_model:
                corrections_found += 1

        # Training data z opravy
        for step_idx in range(len(model_traj) - 1):
            current_point = model_traj[step_idx]
            remaining_points = [
                p for p in all_points_on_page
                if p not in model_traj[:step_idx + 1]
            ]

            if not remaining_points:
                continue

            candidates = find_candidates(
                current_point=current_point,
                remaining_points=remaining_points,
                page_char_scale=page_char_scale,
                num_candidates=num_candidates,
            )

            if not candidates:
                continue

            expert_next_candidates = [
                p for p in expert_trajectory
                if p in candidates and p not in model_traj[:step_idx + 1]
            ]

            if expert_next_candidates:
                expert_choice = expert_next_candidates[0]
                try:
                    expert_action_index = candidates.index(expert_choice)

                    obs_sample = build_observation_from_candidates(
                        current_point=current_point,
                        candidates=candidates,
                        page_char_scale=page_char_scale,
                    )

                    observations.append(obs_sample)
                    actions.append(expert_action_index)
                    total_steps += 1

                except ValueError:
                    pass

    print(f"\nDAgger iterácia {iteration} výsledky:")
    print(f"  Opravy nájdené: {corrections_found}")
    print(f"  Training vzoriek: {total_steps}")

    if total_steps == 0:
        print("Žiadne opravy")
        return None

    observations_arr = np.array(observations, dtype=np.float32)
    actions_arr = np.array(actions, dtype=np.int64)
    infos_arr = np.array([{} for _ in actions_arr])
    dones_arr = np.ones(len(actions_arr), dtype=bool)

    return im_types.Transitions(
        obs=observations_arr,
        acts=actions_arr,
        infos=infos_arr,
        dones=dones_arr,
        next_obs=observations_arr,
    )


# ---------------- Načítanie dát + tréning / načítanie modelu ----------------

expert_data = load_json_data(EXPERT_JSON)
coco_data = load_json_data(COCO_JSON)

env = ReadingEnv(num_candidates=NUM_CANDIDATES)

# ============ TRÉNING PODĽA NASTAVENÍ ============

model = None
transitions = None  # Pre prípad, že budeme potrebovať dáta pre GAIL alebo ďalšie metódy

# ============ TRÉNING BC ============
if TRAIN_BC:
    print("=" * 70)
    print("TRÉNING BC - Behavioral Cloning")
    print("=" * 70)
    print("Preparing data for BC using FINAL logic (contextual candidates + scaling)...")

    transitions = prepare_bc_data(
        expert_data=expert_data,
        coco_data=coco_data,
        num_candidates=NUM_CANDIDATES,
    )

    print(f"Prepared {len(transitions.obs)} training examples.")

    logger = configure(LOG_DIR, ["stdout", "csv", "tensorboard"])

    policy = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_arch=[64, 64],
        lr_schedule=lambda _: 3e-4,
    )

    trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        policy=policy,
        device=DEVICE,
        rng=np.random.default_rng(0),
        custom_logger=logger,
    )

    print("Training BC model...")
    trainer.train(n_epochs=N_EPOCHS)

    torch.save(trainer.policy.state_dict(), BC_MODEL_PATH)
    print(f"Model saved to {BC_MODEL_PATH}")

    model = trainer.policy
else:
    print("=" * 70)
    print("BC TRÉNING VYNECHANÝ (TRAIN_BC = False)")
    print("=" * 70)

# ============ TRÉNING DAGGER ============
if TRAIN_DAGGER:
    # Ak sme nedorobili BC tréning, potrebujeme dáta
    if transitions is None:
        print("\nPreparing data for DAGGER...")
        transitions = prepare_bc_data(
            expert_data=expert_data,
            coco_data=coco_data,
            num_candidates=NUM_CANDIDATES,
        )
    
    # Ak nemáme model, inicializuj ho
    if model is None:
        print("Initializing model for DAGGER...")
        model = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            net_arch=[64, 64],
            lr_schedule=lambda _: 3e-4,
        )
    
    print("\n" + "="*70)
    print("TRÉNING DAGGER - iteratívne zlepšovanie modelu")
    print("="*70)

    # Inicializuj agregovaný dataset s pôvodnými dátami
    aggregated_transitions = transitions

    for dagger_iter in range(1, DAGGER_ITERATIONS + 1):
        # Vygeneruj corrected dáta
        corrected_transitions = dagger_iteration(
            expert_data=expert_data,
            coco_data=coco_data,
            model=model,
            num_candidates=NUM_CANDIDATES,
            iteration=dagger_iter,
        )

        if corrected_transitions is None or len(corrected_transitions.obs) == 0:
            print(f"DAgger iter {dagger_iter}: Žiadne dáta na retréning, končím DAgger.")
            break

        # Kombinuj doteraz agregované dáta s novými opravami
        combined_obs = np.vstack([aggregated_transitions.obs, corrected_transitions.obs])
        combined_acts = np.hstack([aggregated_transitions.acts, corrected_transitions.acts])
        combined_infos = np.hstack([aggregated_transitions.infos, corrected_transitions.infos])
        combined_dones = np.hstack([aggregated_transitions.dones, corrected_transitions.dones])

        # DÔLEŽITÉ: Aktualizuj agregovaný dataset pre ďalšiu iteráciu
        aggregated_transitions = im_types.Transitions(
            obs=combined_obs,
            acts=combined_acts,
            infos=combined_infos,
            dones=combined_dones,
            next_obs=combined_obs,
        )

        # Vytvor nový trainer s novým, väčším datasetom
        logger_dagger = configure(f"{LOG_DIR}/dagger_iter{dagger_iter}", ["stdout"])

        # Vytvoríme novú policy, aby sa stav optimalizátora resetol
        policy_dagger = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            net_arch=[64, 64],
            lr_schedule=lambda _: 5e-4,  # Dočasne vyšší LR na "nakopnutie"
        )
        # Načítaj váhy z predchádzajúceho modelu
        policy_dagger.load_state_dict(model.state_dict())

        trainer_dagger = bc.BC(
            observation_space=env.observation_space,
            action_space=env.action_space,
            demonstrations=aggregated_transitions,
            policy=policy_dagger,
            device=DEVICE,
            rng=np.random.default_rng(dagger_iter),
            custom_logger=logger_dagger,
        )

        print(f"\n  Retraining s {len(aggregated_transitions.obs)} vzorkami...")
        trainer_dagger.train(n_epochs=N_EPOCHS)  # Dôkladnejší tréning na agregovaných dátach

        model = trainer_dagger.policy

        # Vyhodnoť novo-trénaný model
        accuracy_dagger = evaluate_model_accuracy(
            expert_data=expert_data,
            coco_data=coco_data,
            model=model,
            num_candidates=NUM_CANDIDATES,
        )
        print(f"Presnosť po DAgger iterácii {dagger_iter}: {accuracy_dagger:.2f}%")

    # Ulož finálny DAgger model
    torch.save(model.state_dict(), BC_MODEL_PATH.replace(".pt", "_dagger.pt"))
    print(f"\nDAgger model uložený do {BC_MODEL_PATH.replace('.pt', '_dagger.pt')}")
else:
    print("=" * 70)
    print("DAGGER TRÉNING VYNECHANÝ (TRAIN_DAGGER = False)")
    print("=" * 70)

# ============ TRÉNING GAIL ============

if TRAIN_GAIL:
    # Ak nemáme expert dáta, priprav ich
    if transitions is None:
        print("Preparing data for GAIL...")
        transitions = prepare_bc_data(
            expert_data=expert_data,
            coco_data=coco_data,
            num_candidates=NUM_CANDIDATES,
        )
    
    print("\n" + "="*70)
    print("TRÉNING GAIL - Generative Adversarial Imitation Learning (imitation)")
    print("="*70)
    
    from imitation.algorithms.adversarial.gail import GAIL
    from imitation.rewards.reward_nets import BasicRewardNet
    from imitation.util.networks import RunningNorm
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    # Vybudujeme indexy aspoň dočasne pre ENV, aby GAIL vedel roll-outovať po novom
    coco_images_gail, coco_annots_gail = build_coco_index(coco_data)
    file_to_id_gail = {img_info["file_name"]: img_id for img_id, img_info in coco_images_gail.items()}
    
    env.expert_data_ref = expert_data
    env.coco_annots_ref = coco_annots_gail
    env.file_to_id_ref = file_to_id_gail

    venv = DummyVecEnv([lambda: env])
    
    # Skúsime načítať štartovací model, z ktorého GAIL začne
    target_model_path = BC_MODEL_PATH if GAIL_START_MODEL == "bc" else BC_MODEL_PATH.replace(".pt", "_dagger.pt")
    
    learner = PPO(
        "MlpPolicy", 
        venv, 
        n_steps=2048,          # Väčší rollout pre stabilnejšie aktualizácie (viac kontextu naraz)
        batch_size=128,        # Väčší batch pre presnejšie gradienty
        ent_coef=0.001,        # Minimalizovaná explorácia (odstráni zbytočné "náhodné" kroky na konci)
        learning_rate=5e-6,    # Ešte menší LR aby sme si nerozbili to, čo vieme
        clip_range=0.05,       # Extrémne prísne zamedzenie voči zmenám od pôvodnej politiky
        seed=0,
    )

    if os.path.exists(target_model_path):
        print(f"[{GAIL_START_MODEL.upper()}] Načítavam váhy pre PPO generátor zo súboru: {target_model_path}")
        learner.policy.load_state_dict(torch.load(target_model_path, map_location=DEVICE))
    else:
        print(f"[VAROVANIE] Požadovaný model '{target_model_path}' neexistuje. Začínam od nuly/pamäte!")
        if model is not None:
            learner.policy.load_state_dict(model.state_dict())

    reward_net = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    
    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=128,           # Zvýšená vzorka z experta aby lepšie porovnával
        gen_replay_buffer_capacity=2048, 
        n_disc_updates_per_round=1,    
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
        allow_variable_horizon=True,
    )
    
    print("Spúšťam GAIL tréning (extrémne opatrný fine-tuning na dlhšiu dobu)...")
    gail_trainer.train(total_timesteps=100000)
    
    model = learner.policy
    torch.save(model.state_dict(), BC_MODEL_PATH.replace(".pt", "_gail.pt"))
    print(f"\nGAIL model uložený do {BC_MODEL_PATH.replace('.pt', '_gail.pt')}")
else:
    print("=" * 70)
    print("GAIL TRÉNING VYNECHANÝ (TRAIN_GAIL = False)")
    print("=" * 70)

# ============ NAČÍTANIE MODELU (ak nič nebolo natrénované) ============
if model is None:
    if os.path.exists(BC_MODEL_PATH):
        print(f"\nLoading fallback model from {BC_MODEL_PATH}")
        model = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            net_arch=[64, 64],
            lr_schedule=lambda _: 3e-4,
        )
        model.load_state_dict(torch.load(BC_MODEL_PATH, map_location=DEVICE))
        model.eval()

# ---------------- Diagnostika kandidátov ----------------

print("\n" + "="*70)
print("DIAGNOSTIKA: Pokrytie expert_pairs kandidátmi")
print("="*70)

diag_result = diagnose_candidate_coverage(
    expert_data=expert_data,
    coco_data=coco_data,
    num_candidates=NUM_CANDIDATES,
)

print(f"Celkovo expert_pairs: {diag_result['total_pairs']}")
print(f"Expert body V kandidátoch: {diag_result['covered_pairs']}")
print(f"Expert body MIMO kandidátov: {diag_result['uncovered_pairs']}")
print(f"\n[OK] Pokrytie: {diag_result['coverage_percent']:.2f}%")
print(f"[VZDIALENOST] Primerná vzdialenosť k expert bodu (keď nie je v kandidátoch): {diag_result['avg_distance_to_expert']:.2f} px")
print(f"[VZDIALENOST] Max vzdialenosť: {diag_result['max_distance_to_expert']:.2f} px")

if diag_result['uncovered_samples']:
    print(f"\nPríklady NEPOKRYTÝCH expert_pairs (prvých 15):")
    for i, sample in enumerate(diag_result['uncovered_samples'], 1):
        print(f"\n  {i}. [{sample['reason']}] - {sample['img_file']}")
        print(f"     Current: {sample['current_point']}")
        print(f"     Expert bod: {sample['expert_point']}")

        if sample['reason'] == "expert_not_in_candidates":
            closest = sample['closest_candidate']
            dist = sample['distance_to_expert']
            avg_w, avg_h = sample['page_char_scale']
            dist_in_chars = dist / max(avg_w, avg_h)
            print(f"     Najbližší kandidát: {closest}")
            print(f"     Vzdialenosť: {dist:.1f} px ({dist_in_chars:.1f} znakov)")
            print(f"     Počet kandidátov: {sample['num_candidates']}")

print("\n" + "="*70)

if diag_result['coverage_percent'] < 90.0:
    print("[ERROR] PROBLÉM: Menej ako 90% pokrytia - potrebujeme vylepšenie!")
    print("\nNavrhy na vylepšenie:")
    print(f"1. Zvýš NUM_LINES_BELOW (teraz = {NUM_LINES_BELOW}) → skúsiť 4-6")
    print(f"2. Zvýš NUM_CANDIDATES (teraz = {NUM_CANDIDATES}) → skúsiť 15-20")
    print("3. Zväčší y_tolerance pre hľadanie v riadkoch")
elif diag_result['coverage_percent'] < 98.0:
    print("[INFO] PRIJATEĽNÉ: 90-98% pokrytia, ale dá sa vylepšiť")
    print("\nNavrhy: Zvýš NUM_LINES_BELOW alebo NUM_CANDIDATES")
else:
    print("[OK] SKVELÉ: Pokrytie >= 98% - expert body sú dostupní!")
    print("\nAk je presnosť modelu nižšia ako pokrytie, problém je v modeli,")
    print("nie v kandidátoch!")

print("="*70 + "\n")


# ============ FUNKCIE PRE POROVNÁVANIE STRINGOV S RAPIDFUZZ ============

def create_html_diff(model_string: str, expected_string: str, img_file: str, output_dir: str = "diffs") -> None:
    """
    Vytvorí HTML súbor s side-by-side porovnaním model_string a expected_string.
    Používa difflib.HtmlDiff pre vizuálne porovnanie.
    """
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Rozdel stringy na riadky pre lepšie zobrazenie (wrap na 80 znakov)
    def wrap_string(s: str, width: int = 80) -> List[str]:
        return [s[i:i+width] for i in range(0, len(s), width)]
    
    model_lines = wrap_string(model_string)
    expected_lines = wrap_string(expected_string)
    
    # Vytvor HTML diff
    fromdesc = f"Model string ({img_file})"
    todesc = f"Expected string ({img_file})"
    
    html = difflib.HtmlDiff(wrapcolumn=80).make_file(
        model_lines, expected_lines,
        fromdesc=fromdesc,
        todesc=todesc
    )
    
    # Ulož HTML súbor
    base_name = os.path.splitext(img_file)[0]
    output_file = os.path.join(output_dir, f"{base_name}_diff.html")
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(html)
    
    print(f"      Diff HTML: {output_file}")


def trajectory_to_string(
    model_traj: List[Point],
    all_annots_on_page: List[Dict[str, Any]],
    categories: Dict[int, str],
) -> str:
    """
    Konvertuje model trajektóriu (seznam bodov) na string znakov.
    Mapuje body na najbližšie anotácie a zoberá ich labels.
    """
    if not model_traj or not all_annots_on_page:
        return ""

    result_string = ""
    
    for point in model_traj:
        point_arr = np.array(point, dtype=float)
        
        # Nájdi najbližšiu anotáciu k tomuto bodu
        min_distance = float('inf')
        closest_ann = None
        
        for ann in all_annots_on_page:
            bbox = ann["bbox"]  # x, y, w, h
            x, y, w, h = bbox
            cx = x + w / 2.0
            cy = y + h / 2.0
            
            # Apply the same adjustments as in point generation
            label = categories.get(ann["category_id"], "")
            if label == "1":
                cy += h * 0.1
            elif label == "6":
                cx -= w * 0.15
                cy += h * 0.15
            elif label == "9":
                cx += w * 0.15
                cy -= h * 0.2
            
            # Vypočítaj vzdialenosť
            ann_center = np.array([cx, cy], dtype=float)
            distance = np.linalg.norm(point_arr - ann_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_ann = ann
        
        # Ak máme najbližšiu anotáciu, pridaj jej label do stringu
        if closest_ann is not None:
            label: str = categories.get(closest_ann["category_id"], "")
            result_string += label
    
    return result_string


def load_expected_string(img_file: str) -> Optional[str]:
    """Načíta očakávaný string znakov pre obrázok."""
    strings_dir = "image_strings"
    
    # Zmeň príponu z .jpg na .txt
    base_name = os.path.splitext(img_file)[0]
    string_file = os.path.join(strings_dir, base_name + ".txt")
    
    if not os.path.exists(string_file):
        return None
    
    try:
        with open(string_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading {string_file}: {e}")
        return None


def compare_strings_with_rapidfuzz(
    model_string: str,
    expected_string: str,
) -> Dict[str, Any]:
    """
    Porovná model string s očakávaným stringom pomocou rapidfuzz Levenshtein distance a similarity.
    Vracia Levenshtein distance a similarity ratio.
    """
    if not RAPIDFUZZ_AVAILABLE:
        return {
            "error": "rapidfuzz not available",
            "levenshtein_distance": 0.0,
            "similarity": 0.0,
        }
    
    lev_dist = Levenshtein.distance(model_string, expected_string)
    max_len = max(len(model_string), len(expected_string))
    if max_len == 0:
        similarity = 100.0
    else:
        similarity = (1 - (lev_dist / max_len)) * 100
    
    return {
        "levenshtein_distance": lev_dist,
        "similarity": similarity,
        "model_length": len(model_string),
        "expected_length": len(expected_string),
    }


print("\n" + "="*70)
print("Tréning dokončený!")
print("="*70)



# ---------------- Generovanie a kreslenie trajektórií ----------------

print("\nGenerating and evaluating trajectories for all available models...")

categories: Dict[int, str] = {
    c["id"]: c["name"] for c in coco_data.get("categories", [])
}
coco_images, coco_annots = build_coco_index(coco_data)

file_to_id: Dict[str, int] = {
    img_info["file_name"]: img_id
    for img_id, img_info in coco_images.items()
}

models_to_evaluate = {
    "BC": BC_MODEL_PATH,
    "DAgger": BC_MODEL_PATH.replace(".pt", "_dagger.pt"),
    "GAIL": BC_MODEL_PATH.replace(".pt", "_gail.pt")
}

all_model_results = {}

for model_name, model_path in models_to_evaluate.items():
    if not os.path.exists(model_path):
        print(f"[{model_name}] Model nenájdený na ceste: {model_path}. Preskakujem vyhodnotenie.")
        continue

    print(f"\n[{model_name}] Načítavam a vyhodnocujem model...")
    eval_model = ActorCriticPolicy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        net_arch=[64, 64],
        lr_schedule=lambda _: 3e-4,
    )
    eval_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    eval_model.eval()

    comparison_results = []
    
    outdir_model = f"{OUTDIR}_{model_name.lower()}"
    diff_dir_model = f"diffs_{model_name.lower()}"
    os.makedirs(outdir_model, exist_ok=True)
    
    for item in expert_data:
        img_file = item.get("image_file")
        if not img_file:
            continue

        img_id = file_to_id.get(img_file)
        if img_id is None:
            continue

        all_annots_on_page = coco_annots.get(img_id, [])

        all_points: List[Point] = []
        for ann in all_annots_on_page:
            bbox: BBox = tuple(ann["bbox"])
            label = categories.get(ann["category_id"], "")

            x, y, w, h = bbox
            cx = x + w / 2.0
            cy = y + h / 2.0

            if label == "1":
                cy += h * 0.1
            elif label == "6":
                cx -= w * 0.15
                cy += h * 0.15
            elif label == "9":
                cx += w * 0.15
                cy -= h * 0.2
            all_points.append((cx, cy))

        obs, _ = env.reset(options={"all_points": all_points, "annotations": all_annots_on_page})

        model_traj: List[Point] = []
        if env.current_point is not None:
            model_traj.append(env.current_point)

        done = False
        while not done:
            action, _ = eval_model.predict(obs, deterministic=True)
            obs, _, done, _, _ = env.step(int(action))
            if env.current_point is not None:
                model_traj.append(env.current_point)

        # Porovnanie
        model_string = trajectory_to_string(model_traj, all_annots_on_page, categories)
        expected_string = load_expected_string(img_file)
        
        if expected_string:
            metrics = compare_strings_with_rapidfuzz(model_string, expected_string)
            comparison_results.append({
                "image_file": img_file,
                "model_string": model_string,
                "expected_string": expected_string,
                "metrics": metrics,
            })
            create_html_diff(model_string, expected_string, img_file, output_dir=diff_dir_model)
        
        # Kreslenie
        img_path = os.path.join(IMAGE_DIR, img_file)
        out_path = os.path.join(outdir_model, img_file)

        img = cv2.imread(img_path)
        if img is None:
            img = 255 * np.ones((2000, 2000, 3), dtype=np.uint8)

        for pair in item.get("expert_pairs", []):
            p_start = tuple(map(int, pair["state"]))
            p_end = tuple(map(int, pair["action"]))
            cv2.arrowedLine(img, p_start, p_end, color=(0, 0, 255), thickness=3, tipLength=0)

        for i in range(len(model_traj) - 1):
            p0 = tuple(map(int, model_traj[i]))
            p1 = tuple(map(int, model_traj[i + 1]))
            cv2.line(img, p0, p1, color=(0, 255, 0), thickness=2)

        cv2.imwrite(out_path, img)
    
    all_model_results[model_name] = comparison_results


# ========== FINÁLNY REPORT ==========

print("\n" + "="*70)
print("FINÁLNY REPORT - POROVNANIE MODELOV (BC vs DAgger vs GAIL)")
print("="*70 + "\n")

summary_table = []

for model_name, results in all_model_results.items():
    if not results:
        continue
    
    lev_distances = []
    similarities = []
    for r in results:
        metrics = r["metrics"]
        if "error" not in metrics:
            lev_distances.append(metrics["levenshtein_distance"])
            similarities.append(metrics["similarity"])
    
    if lev_distances:
        avg_lev = np.mean(lev_distances)
        avg_sim = np.mean(similarities)
        summary_table.append(f"{model_name:<10} | {avg_lev:>10.2f} | {avg_sim:>10.2f}%")

print(f"{'Model':<10} | {'Avg L. Dist':>10} | {'Avg Similarity':>10}")
print("-" * 38)
for row in summary_table:
    print(row)

print("\n" + "="*70)
print("Odložené HTML aj obrázky obdržali separátne adresáre (napr. image_bc_traj_bc, diffs_dagger, atď.)")
print("Done.")
print("="*70)
