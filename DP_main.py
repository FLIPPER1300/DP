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

from dp_core import (
    Point, BBox, NUM_CANDIDATES, NUM_LINES_BELOW, 
    load_json_data, compute_page_char_scale, build_coco_index, 
    find_candidates, ReadingEnv, BaseDataLoader
)

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

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

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
    is_yolo: bool = False
) -> str:
    """
    Konvertuje model trajektóriu (seznam bodov) na string znakov.
    Mapuje body na najbližšie anotácie a zoberie ich labels.
    """
    if not model_traj or not all_annots_on_page:
        return ""

    result_string = ""
    
    for point in model_traj:
        point_arr = np.array(point, dtype=float)
        
        # Nájdí najbližšiu anotáciu k tomuto bodu
        min_distance = float('inf')
        closest_ann = None
        
        for ann in all_annots_on_page:
            if is_yolo:
                # Pre YOLO získame priamo string z YOLO triedy, nie z COCO mappingu, aby sme predišli nesúladu ID.
                label = ann.get("yolo_label", str(ann["category_id"]))
                cx, cy = ann["centroid"]
            else:
                label = categories.get(ann["category_id"], "")
                cx, cy = ann["centroid"]
            
            # Vypočítaj vzdialenosť
            ann_center = np.array([cx, cy], dtype=float)
            distance = np.linalg.norm(point_arr - ann_center)
            
            if distance < min_distance:
                min_distance = distance
                closest_ann = ann
        
        # Ak máme najbližšiu anotáciu, pridaj jej label do stringu
        if closest_ann is not None:
            if is_yolo:
                label = closest_ann.get("yolo_label", str(closest_ann["category_id"]))
            else:
                label = categories.get(closest_ann["category_id"], "")
            result_string += str(label)
    
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

# Nacitanie YOLO detekcii
YOLO_JSON = "yolo_detections.json"
yolo_data = load_json_data(YOLO_JSON) if os.path.exists(YOLO_JSON) else {}

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
    for data_mode in ["COCO", "YOLO"]:
        if data_mode == "YOLO" and not yolo_data:
            print("[YOLO] Data sa nenašli, preskakujem YOLO evaluáciu.")
            continue
            
        print(f"[{model_name}] Beh na dátach: {data_mode}")
        
        # Set up result structures
        eval_model = ActorCriticPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            net_arch=[64, 64],
            lr_schedule=lambda _: 3e-4,
        )
        eval_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        eval_model.eval()

        comparison_results = []
        
        result_key = f"{model_name}" if data_mode == "COCO" else f"{model_name} (YOLO)"
        outdir_model = f"{OUTDIR}_{model_name.lower()}" if data_mode == "COCO" else f"{OUTDIR}_{model_name.lower()}_yolo"
        diff_dir_model = f"diffs_{model_name.lower()}" if data_mode == "COCO" else f"diffs_{model_name.lower()}_yolo"
        os.makedirs(outdir_model, exist_ok=True)
        os.makedirs(diff_dir_model, exist_ok=True)
        
        for item in expert_data:
            img_file = item.get("image_file")
            if not img_file:
                continue

            all_points: List[Point] = []
            all_annots_on_page: List[Dict[str, Any]] = []

            if data_mode == "COCO":
                img_id = file_to_id.get(img_file)
                if img_id is None:
                    continue
                all_annots_on_page = coco_annots.get(img_id, [])
                all_points, all_annots_on_page = BaseDataLoader.parse_coco(all_annots_on_page, categories)
            else:
                # YOLO Data Loading pomocou BaseDataLoader
                yolo_annots = yolo_data.get(img_file, [])
                if not yolo_annots:
                    continue
                # Vráti [((cx, cy)), ...], [{"category_id": x, ...}, ...]
                all_points, all_annots_on_page = BaseDataLoader.parse_yolo(yolo_annots, confidence_threshold=0.0)

            if not all_points:
                continue

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
            is_yolo = (data_mode == "YOLO")
            model_string = trajectory_to_string(model_traj, all_annots_on_page, categories, is_yolo=is_yolo)
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
        
        all_model_results[result_key] = comparison_results


# ========== FINÁLNY REPORT ==========

print("\n" + "="*70)
print("FINÁLNY REPORT - POROVNANIE MODELOV (BC vs DAgger vs GAIL) + YOLO verianty")
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
        summary_table.append(f"{model_name:<16}  {avg_lev:>10.2f}  {avg_sim:>10.2f}%")

print(f"{'Model':<16}  {'Avg L. Dist':>10}  {'Avg Similarity':>10}")
print("-" * 44)
for row in summary_table:
    print(row)

print("\n" + "="*70)
print("Odložené HTML aj obrázky obdržali separátne adresáre (napr. image_bc_traj_bc, diffs_dagger, atď.)")
print("Done.")
print("="*70)