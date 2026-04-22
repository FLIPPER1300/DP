import os
import json
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction

def sort_detections(detections):
    if not detections:
        return []
    
    # Najprv zoradiť podľa y
    detections.sort(key=lambda d: d["centroid"][1])
    
    rows = []
    current_row = [detections[0]]
    y_threshold = 20  # Prah pre združovanie do rovnakého riadku (pixelov)
    
    for det in detections[1:]:
        if abs(det["centroid"][1] - current_row[0]["centroid"][1]) < y_threshold:
            current_row.append(det)
        else:
            rows.append(current_row)
            current_row = [det]
    if current_row:
        rows.append(current_row)
        
    # V každom riadku zoradiť podľa x
    sorted_detections = []
    for row in rows:
        row.sort(key=lambda d: d["centroid"][0])
        sorted_detections.extend(row)
        
    return sorted_detections

def main():
    model_path = 'yolo_v8_digits_best.pt'
    images_dir = 'images/'
    output_json = 'yolo_detections.json'
    
    # Nastavíme zariadenie a confidence threshold podľa požiadavky
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading YOLOv8 model on {device}...")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=model_path,
        confidence_threshold=0.6, # Pridaný confidence threshold (dá sa upraviť)
        device=device,
    )

    all_results = {}
    
    if not os.path.exists(images_dir):
        print(f"Directory {images_dir} does not exist.")
        return

    # Súčasťou optimalizácie SAHI pre rýchlosť bez veľkej straty presnosti:
    # 1. Zväčšíme slice_width a slice_height na (väčšie úseky modelu zaberú menej iterácií)
    # 2. Zmenšíme prekrývanie (overlap) na menšie percento (napr. 0.1)
    # 3. NMS (Non-Maximum Suppression) je už v SAHI integrované pri default params.
    
    for img_name in os.listdir(images_dir):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
            
        img_path = os.path.join(images_dir, img_name)
        
        # Sliced prediction
        result = get_sliced_prediction(
            img_path,
            detection_model,
            slice_height=512, # Zmenšené pre lepšiu detekciu malých znakov
            slice_width=512,
            overlap_height_ratio=0.2, # Zväčšené prekrytie, aby sa nestrácali znaky na okrajoch
            overlap_width_ratio=0.2,
            postprocess_type='NMS', # Vynútené NMS ako postprocessing v SAHI
            postprocess_match_metric='IOU',
            postprocess_match_threshold=0.3 # NMS IOU threshold (vymazáva duplikáty)
        )
        
        image_detections = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox
            
            w = bbox.maxx - bbox.minx
            h = bbox.maxy - bbox.miny
            
            # Výpočet centroidu s offestmi z DP_main.py
            x_center = bbox.minx + w / 2.0
            y_center = bbox.miny + h / 2.0
            
            # Názov kategórie (môže byť '1', '6', '9' atď. podľa natrénovaných tried YOLO modelu)
            label = str(obj.category.name)
            # V prípade ak sú triedy iba idéčka, skontrolujeme aj ID
            label_id = str(obj.category.id)
            
            if label == "1" or label_id == "1":
                y_center += h * 0.1
            elif label == "6" or label_id == "6":
                x_center -= w * 0.15
                y_center += h * 0.15
            elif label == "9" or label_id == "9":
                x_center += w * 0.15
                y_center -= h * 0.2
            
            image_detections.append({
                "class": obj.category.id,
                "centroid": [x_center, y_center],
                "bbox": [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy]
            })
            
        # Zoradíme zhora nadol a zľava doprava
        sorted_dets = sort_detections(image_detections)
        
        # Vyčistenie nepotrebných dát
        final_list = []
        for det in sorted_dets:
            final_list.append({
                "class": det["class"],
                "centroid": det["centroid"],
                "bbox": det["bbox"]
            })
            
        all_results[img_name] = final_list
        print(f"Processed {img_name} - Found {len(final_list)} objects")

    with open(output_json, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Results saved to {output_json}")

if __name__ == "__main__":
    main()
