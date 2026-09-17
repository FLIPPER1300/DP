# Imitation Learning for Transcription of Historical Encrypted Manuscripts

Master's thesis project developing and evaluating an algorithm for the **automated transcription of historical encrypted (ciphered) manuscripts** into machine-readable text. The core idea is to treat "reading" a line of a manuscript as a **sequential decision-making problem**: given the detected symbols on a page (as points / bounding boxes), an agent learns to pick the order in which to visit them — imitating how a human expert reads the document — and outputs the resulting character string.

This work is part of the research project Artificial Intelligence for Encrypted Handwritten Document Processing, project code `09I05-03-V02-00031`.

> **Note:** This repository contains the source code only. The full image dataset and some trained model weights are not included (see [Data & Models](#data--models) below).

## Overview

1. **Symbol detection** — a YOLOv8 object detector (run via [SAHI](https://github.com/obss/sahi) sliced inference for small symbols on large scanned pages) locates individual cipher symbols/digits in manuscript images and outputs their bounding boxes and centroids.
2. **Expert trajectory construction** — ground-truth COCO-style annotations are converted into ordered "expert trajectories": the sequence in which a human transcriber would move across symbols on a page, line by line.
3. **Reading environment** — a custom [Gymnasium](https://gymnasium.farama.org/) environment (`ReadingEnv`) models one reading step as choosing the next symbol to visit from a shortlist of nearby candidate symbols (same line, following lines, or nearest remaining point).
4. **Imitation learning** — a policy is trained to imitate the expert's reading order using three approaches, in increasing order of sophistication:
   - **Behavioral Cloning (BC)** — supervised learning directly on expert (state, action) pairs.
   - **DAgger (Dataset Aggregation)** — iteratively rolls out the current policy, compares it against the expert, and aggregates corrections into the training set.
   - **GAIL (Generative Adversarial Imitation Learning)** — adversarial fine-tuning (PPO generator + learned reward/discriminator) starting from the BC or DAgger policy.
5. **Transcription & evaluation** — the trained policy walks across the detected symbols of a page, producing an ordered character string, which is compared against the human ground-truth transcription using Levenshtein distance / similarity (via `rapidfuzz`) and visualized as an HTML diff.
6. **Web application** — a Flask + React demo app lets a user upload a manuscript image, pick a YOLO model and an imitation-learning model, and view the detected symbols, the predicted reading trajectory, the transcribed string, and (optionally) a diff against an expected transcription.

## Pipeline

```
_annotations.coco.json ─┐
                         ├─► generate_expert_trajectories.py ─► expert_trajectories(_repaired).json
   images/               │                                            │
                         │                                            ├─► generate_string_from_expert.py ─► image_strings/*.txt
                         │                                            │      (ground-truth transcription strings)
                         │                                            │
                         └─► generate_json_data_from_yolo.py          └─► used as expert data for training
              (YOLOv8 + SAHI detections) ─► yolo_detections.json

DP_main.py
  ├─ loads expert_trajectories + COCO annotations
  ├─ trains BC / DAgger / GAIL policies (dp_core.ReadingEnv)
  ├─ evaluates each trained policy on COCO ground-truth points AND on YOLO-detected points
  └─ writes: trajectory visualizations, HTML diffs, and a summary accuracy table
```

`generate_json_data_for_images.py` is a utility that dumps the COCO annotations into one small per-image JSON file each (centroid + label + bbox), useful for inspection/debugging outside the main pipeline.

## Repository Structure

```
DP_main.py                        Main training & evaluation script (BC, DAgger, GAIL)
dp_core.py                        Shared core: ReadingEnv (Gymnasium env), candidate search,
                                   COCO/YOLO data loaders, observation building
generate_expert_trajectories.py   Builds expert reading trajectories from COCO annotations
                                   (row detection via iterative line fitting) + visualization
generate_string_from_expert.py    Extracts ground-truth transcription strings (image_strings/*.txt)
                                   from expert trajectories
generate_json_data_from_yolo.py   Runs YOLOv8 (+SAHI sliced inference) over a folder of images
                                   to produce yolo_detections.json
generate_json_data_for_images.py  Dumps per-image annotation JSON files from the COCO file
_annotations.coco.json            COCO-format ground-truth symbol annotations (dataset)
expert_trajectories.json /
expert_trajectories_repaired.json Precomputed expert trajectories derived from the annotations

web_app/
├── backend/
│   ├── app.py                    Flask API: YOLO detection + imitation-model inference,
│   │                              trajectory drawing, string comparison/diff endpoints
│   └── requirements.txt          Backend Python dependencies
└── frontend/
    ├── src/App.js, Modal.js      React UI: upload image, pick models, view results & diff
    └── package.json              Frontend dependencies (React 18, axios)
```

## Method Details

### Reading environment (`dp_core.ReadingEnv`)

- **State**: the current symbol's position plus a fixed-size list of candidate next symbols (`NUM_CANDIDATES = 15`), found by `find_candidates()`:
  1. remaining symbols to the right on the same line,
  2. symbols on the next `NUM_LINES_BELOW` (= 3) lines below,
  3. otherwise, the nearest remaining symbols by Euclidean distance.
- **Observation**: for each candidate, its position relative to the current point, normalized by the page's average character width/height.
- **Action**: a discrete choice among the (padded) candidate list — which candidate to move to next.
- Works interchangeably with **COCO ground-truth** annotations or **YOLO-detected** symbols (`BaseDataLoader.parse_coco` / `parse_yolo`), including label-specific centroid offset corrections for visually ambiguous digits (e.g. `1`, `6`, `9`).

### Training (`DP_main.py`)

Training stages are toggled independently via flags at the top of the script:

```python
TRAIN_BC = False
TRAIN_DAGGER = False
TRAIN_GAIL = True
GAIL_START_MODEL = "dagger"   # which checkpoint GAIL fine-tunes from: "bc" or "dagger"
N_EPOCHS = 50
DAGGER_ITERATIONS = 2
```

- **BC** uses `imitation.algorithms.bc` with an `ActorCriticPolicy` (`stable_baselines3`), trained on `Transitions` built from expert (state → chosen candidate) pairs.
- **DAgger** repeatedly rolls the current policy out through `ReadingEnv`, finds where it diverges from the expert trajectory, adds corrective (state, expert action) pairs, and retrains on the aggregated dataset.
- **GAIL** (`imitation.algorithms.adversarial.gail`) wraps `ReadingEnv` in a `DummyVecEnv` and fine-tunes a PPO policy against a learned reward network, initialized from the BC or DAgger weights.

Trained weights are saved as `bc_model.pt`, `bc_model_dagger.pt`, `bc_model_gail.pt`.

### Evaluation

For every trained model found on disk, `DP_main.py`:
1. Rolls the policy across each page's symbols (both COCO ground-truth and, if available, YOLO-detected points).
2. Converts the resulting trajectory into a transcribed character string (`trajectory_to_string`).
3. Compares it to the ground-truth string from `image_strings/` using Levenshtein distance and similarity (`rapidfuzz`).
4. Writes an HTML side-by-side diff (`diffs_<model>/`) and a trajectory visualization image (`image_bc_traj_<model>/`), and prints a final summary table (average distance / similarity per model).

## Web Application

A demo app for interactively transcribing a single uploaded manuscript image.

- **Backend** (`web_app/backend/app.py`, Flask): runs YOLO + SAHI detection on the uploaded image, feeds the detected symbols into a chosen trained imitation-learning policy (`ReadingEnv`), draws the detected bounding boxes and predicted trajectory, and (if an expected transcription is provided) returns a Levenshtein accuracy score and HTML diff. Falls back to simple top-left-to-bottom-right sorting if RL inference fails.
- **Frontend** (`web_app/frontend`, React): lets the user pick a YOLO model and an imitation-learning model (auto-discovered from `.pt` files in the repo root), upload an image, optionally paste an expected transcription, and view the annotated image, the predicted trajectory, the transcribed string, and the diff.

### Running the web app

```bash
# Backend (from web_app/backend)
pip install -r requirements.txt
python app.py            # serves API on http://localhost:5000

# Frontend (from web_app/frontend, in a separate terminal)
npm install
npm start                 # serves UI on http://localhost:3000
```

The backend auto-discovers YOLO weights (`*.pt` with "yolo" in the filename) and imitation-learning weights (`*.pt` with "bc" or "dagger" in the filename) placed in the repository root.

## Setup

Requirements are split across the training pipeline and the web app.

**Training / evaluation pipeline** (`DP_main.py`, `dp_core.py`, `generate_*.py`) needs, at minimum:

```bash
pip install torch torchvision torchaudio numpy opencv-python gymnasium \
            stable-baselines3 imitation sahi ultralytics rapidfuzz
```

- Optional: `torch-directml` for AMD GPU acceleration on Windows (falls back to CUDA, then CPU, automatically).

**Web app backend**: see [web_app/backend/requirements.txt](web_app/backend/requirements.txt).

**Web app frontend**: see [web_app/frontend/package.json](web_app/frontend/package.json) (Node.js + npm required).

## Data & Models

This repository does **not** include the full manuscript image dataset or all trained model checkpoints — only the code and the derived JSON annotation/trajectory files. To reproduce the full pipeline you need:

- `images/` — the manuscript page images referenced by `_annotations.coco.json` and `expert_trajectories*.json`.
- A trained YOLOv8 symbol-detection model (e.g. `yolo_v8_digits_best.pt`) for `generate_json_data_from_yolo.py` and the web app.
- Trained imitation-learning checkpoints (`bc_model.pt`, `bc_model_dagger.pt`, `bc_model_gail.pt`) produced by running `DP_main.py`, for evaluation and the web app.

## Typical Workflow

```bash
# 1. (Optional) Regenerate expert trajectories from the COCO annotations
python generate_expert_trajectories.py

# 2. (Optional) Extract ground-truth transcription strings for evaluation
python generate_string_from_expert.py

# 3. (Optional) Run YOLO detection over a folder of images
python generate_json_data_from_yolo.py

# 4. Train BC / DAgger / GAIL policies and evaluate them
#    (toggle TRAIN_BC / TRAIN_DAGGER / TRAIN_GAIL flags at the top of the file)
python DP_main.py
```

## Evaluation Metric

Transcription accuracy is measured by comparing the model-generated character string to the manually transcribed ground truth using **Levenshtein edit distance** and a normalized **similarity ratio** (`rapidfuzz.distance.Levenshtein`), reported per document and averaged per model in the final summary table.
