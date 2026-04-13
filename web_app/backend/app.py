from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import cv2
import torch
import numpy as np
from sahi.slicing import slice_image
from ultralytics import YOLO
from PIL import Image
import json
from collections import defaultdict
from torchvision.ops import nms
from rapidfuzz.distance import Levenshtein
import difflib

app = Flask(__name__)
CORS(app)

# --- Model a dátové cesty ---
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
YOLO_MODELS_DIR = ROOT_DIR
IMITATION_MODELS_DIR = ROOT_DIR

yolo_models_available = [f for f in os.listdir(YOLO_MODELS_DIR) if f.endswith('.pt') and 'yolo' in f]
imitation_models_available = [f for f in os.listdir(IMITATION_MODELS_DIR) if f.endswith('.pt') and ('bc' in f or 'dagger' in f)]

UPLOADS_DIR = os.path.join(os.path.dirname(__file__), 'uploads')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Helper funkcie ---

def get_bounding_boxes_from_image(image_path, model_path):
    """Spracuje obrázok, rozreže ho a vráti bounding boxy."""
    yolo_model = YOLO(model_path)
    slice_height = 512
    slice_width = 512
    overlap_height_ratio = 0.2
    overlap_width_ratio = 0.2

    sliced_image_result = slice_image(
        image=image_path,
        slice_height=slice_height,
        slice_width=slice_width,
        overlap_height_ratio=overlap_height_ratio,
        overlap_width_ratio=overlap_width_ratio,
    )

    all_boxes = []
    all_scores = []
    all_labels = []

    for i, sliced_image in enumerate(sliced_image_result.images):
        image_pil = Image.fromarray(sliced_image)
        results = yolo_model(image_pil)

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                score = box.conf[0].item()
                label_index = int(box.cls[0].item())
                label = yolo_model.names[label_index]

                original_x1 = x1 + sliced_image_result.starting_pixels[i][0]
                original_y1 = y1 + sliced_image_result.starting_pixels[i][1]
                original_x2 = x2 + sliced_image_result.starting_pixels[i][0]
                original_y2 = y2 + sliced_image_result.starting_pixels[i][1]

                all_boxes.append([original_x1, original_y1, original_x2, original_y2])
                all_scores.append(score)
                all_labels.append(label)

    return all_boxes, all_scores, all_labels

def non_max_suppression(boxes, scores, labels, conf_threshold=0.5, iou_threshold=0.3):
    """Potlačenie nemaximálnych hodnôt pre bounding boxy pomocou torchvision.ops.nms."""
    if not boxes:
        return [], [], []

    # Filter by confidence threshold
    keep = [i for i, score in enumerate(scores) if score > conf_threshold]
    boxes = [boxes[i] for i in keep]
    scores = [scores[i] for i in keep]
    labels = [labels[i] for i in keep]

    if not boxes:
        return [], [], []

    boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
    scores_tensor = torch.tensor(scores, dtype=torch.float32)
    
    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    
    keep_boxes = [boxes[i] for i in keep_indices]
    keep_scores = [scores[i] for i in keep_indices]
    keep_labels = [labels[i] for i in keep_indices]
    
    return keep_boxes, keep_scores, keep_labels


def draw_bounding_boxes(image_path, boxes, labels, output_path):
    """Vykreslí bounding boxy do obrázka."""
    image = cv2.imread(image_path)
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, str(label), (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.imwrite(output_path, image)

def calculate_centroids_with_offsets(boxes, labels):
    """Vypočíta centroidy a aplikuje špecifické posuny pre duplicitné znaky."""
    centroids = []
    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w / 2, y1 + h / 2

        if label == "1":
            cy += h * 0.1
        elif label == "6":
            cx -= w * 0.15
            cy += h * 0.15
        elif label == "9":
            cx += w * 0.15
            cy -= h * 0.2
        
        centroids.append((cx, cy))
    return centroids

def calculate_avg_char_height(boxes):
    """Vypočíta priemernú výšku znakov z bounding boxov."""
    if not boxes:
        return 100  # Default fallback
    heights = [y2 - y1 for x1, y1, x2, y2 in boxes]
    return np.mean(heights) if heights else 100

def sort_centroids(centroids, labels, avg_char_height=100):
    """Zoradí centroidy a label-y zľava doprava, zhora nadol, s ohľadom na duplicity.
    
    Riadky sú určené na základe priemernej výšky znakov.
    """
    if not centroids:
        return [], []

    # Použijeme defaultdict na sledovanie, koľkokrát sme už videli daný centroid
    centroid_counts = defaultdict(int)
    adjusted_centroids = []
    for (cx, cy) in centroids:
        # Mierne upravíme pozíciu, aby sme sa vyhli presným duplikátom pri triedení
        offset = centroid_counts[(cx, cy)] * 1e-5
        adjusted_centroids.append((cx + offset, cy + offset))
        centroid_counts[(cx, cy)] += 1

    # Zoradenie na základe upravených centroidov s dynamickým thresholdom (1.5x výška znaku)
    row_threshold = avg_char_height * 1.5
    combined = sorted(zip(adjusted_centroids, labels, centroids), 
                     key=lambda item: (item[0][1] // row_threshold, item[0][0]))
    
    # Vrátime pôvodné (neupravené) centroidy v správnom poradí
    _, sorted_labels, original_sorted_centroids = zip(*combined)
    
    return list(original_sorted_centroids), list(sorted_labels)


def draw_trajectory(image_path, trajectory, labels, output_path):
    """Vykreslí trajektóriu do obrázka."""
    image = cv2.imread(image_path)
    for i in range(len(trajectory) - 1):
        pt1 = tuple(map(int, trajectory[i]))
        pt2 = tuple(map(int, trajectory[i+1]))
        cv2.line(image, pt1, pt2, (255, 0, 0), 2)

    for i, point in enumerate(trajectory):
        pt = tuple(map(int, point))
        cv2.circle(image, pt, 5, (0, 0, 255), -1)
        cv2.putText(image, str(labels[i]), (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    cv2.imwrite(output_path, image)

def levenshtein_distance_and_accuracy(s1, s2):
    if len(s1) == 0 and len(s2) == 0: 
        return 0, 100.0
    
    dist = Levenshtein.distance(s1, s2)
    accuracy = Levenshtein.normalized_similarity(s1, s2) * 100.0
    return dist, accuracy

def wrap_string(s, width=80):
    return [s[i:i+width] for i in range(0, len(s), width)]

# --- API Endpoints ---

@app.route('/api/models', methods=['GET'])
def get_models():
    return jsonify({
        'yolo_models': yolo_models_available,
        'imitation_models': imitation_models_available
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    yolo_model_name = request.form.get('yolo_model')
    imitation_model_name = request.form.get('imitation_model')
    expected_string = request.form.get('expected_string', '').strip()

    if not yolo_model_name or not imitation_model_name:
        return jsonify({'error': 'Model not specified'}), 400

    yolo_model_path = os.path.join(YOLO_MODELS_DIR, yolo_model_name)
    imitation_model_path = os.path.join(IMITATION_MODELS_DIR, imitation_model_name)

    if not os.path.exists(yolo_model_path) or not os.path.exists(imitation_model_path):
        return jsonify({'error': 'Model file not found'}), 404

    try:
        filename = file.filename
        filepath = os.path.join(UPLOADS_DIR, filename)
        file.save(filepath)

        # 1. YOLO Detekcia
        boxes, scores, labels = get_bounding_boxes_from_image(filepath, yolo_model_path)
        boxes, scores, labels = non_max_suppression(boxes, scores, labels)

        # Vykreslenie bounding boxov
        processed_image_filename = f"processed_{filename}"
        processed_image_path = os.path.join(RESULTS_DIR, processed_image_filename)
        draw_bounding_boxes(filepath, boxes, labels, processed_image_path)

        # 2. Príprava dát pre imitačný model
        avg_char_height = calculate_avg_char_height(boxes)
        centroids = calculate_centroids_with_offsets(boxes, labels)
        sorted_centroids, sorted_labels = sort_centroids(centroids, labels, avg_char_height)
        
        # 3. Imitačný model (Behavioral Cloning)
        # V tejto zjednodušenej verzii len prechádzame cez zoradené centroidy
        trajectory_points = sorted_centroids
        trajectory_labels = sorted_labels
        
        # Vykreslenie trajektórie
        trajectory_image_filename = f"trajectory_{filename}"
        trajectory_image_path = os.path.join(RESULTS_DIR, trajectory_image_filename)
        draw_trajectory(filepath, trajectory_points, trajectory_labels, trajectory_image_path)

        # 4. Vytvorenie stringu
        detected_string = "".join(map(str, trajectory_labels))

        response_data = {
            'processed_image_url': f'/results/{processed_image_filename}',
            'trajectory_image_url': f'/results/{trajectory_image_filename}',
            'detected_string': detected_string
        }

        if expected_string:
            distance, accuracy = levenshtein_distance_and_accuracy(expected_string, detected_string)
            
            # Použi difflib.HtmlDiff rovnako ako v DP_main.py
            expected_lines = wrap_string(expected_string, width=80)
            detected_lines = wrap_string(detected_string, width=80)
            
            fromdesc = "Expected string"
            todesc = "Detected string"
            
            diff_table = difflib.HtmlDiff(wrapcolumn=80).make_table(
                expected_lines, detected_lines,
                fromdesc=fromdesc,
                todesc=todesc
            )

            diff_css = """
            <style type="text/css">
                table.diff {font-family:Courier; border-collapse: collapse; margin: 10px auto 0 auto; width: auto; min-width: 80%; text-align: left;}
                table.diff th, table.diff td {padding: 5px; border: 1px solid #ccc;}
                .diff_header {background-color:#e0e0e0}
                td.diff_header {text-align:right; width: 5%;}
                .diff_next {background-color:#c0c0c0; width: 5%;}
                .diff_add {background-color:#aaffaa}
                .diff_chg {background-color:#ffff77}
                .diff_sub {background-color:#ffaaaa}
            </style>
            """
            
            diff_html = f"""
            <div style="width: 100%; overflow-x: auto; text-align: center;">
                {diff_css}
                {diff_table}
            </div>
            """

            response_data['expected_string'] = expected_string
            response_data['levenshtein_distance'] = distance
            response_data['accuracy'] = round(accuracy, 2)
            response_data['diff_html'] = diff_html

        return jsonify(response_data)

    except Exception as e:
        app.logger.error(f"Error processing file: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/results/<path:filename>')
def get_result_image(filename):
    return send_from_directory(RESULTS_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
