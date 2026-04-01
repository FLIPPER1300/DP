import json
import os

def generate_string_files():
    """
    Nacita expert_trajectories_repaired.json, extrahuje expert_pairs
    a ulozi stringy cislic (labels) ako textove subory do priecinka image_strings.
    expert_pairs obsahuju poradie centroidov, z ktorych sa beru labels.
    """
    
    # Konfiguracia
    json_file = "expert_trajectories_repaired.json"
    output_dir = "image_strings"
    
    # Vytvor priecink, ak neexistuje
    os.makedirs(output_dir, exist_ok=True)
    
    # Nacitaj JSON
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Spracuj kazdy zaznam
    for entry in data:
        image_file = entry.get("image_file", "")
        trajectory = entry.get("trajectory", [])
        expert_pairs = entry.get("expert_pairs", [])
        
        if not image_file or not expert_pairs:
            print(f"[!] Preskakujem {image_file} - ziadne expert_pairs")
            continue
        
        # Vytvor nazov textoveho suboru
        file_name = os.path.splitext(image_file)[0] + ".txt"
        file_path = os.path.join(output_dir, file_name)
        
        # Vytvor mapping centroid -> label z trajectory
        trajectory_map = {}
        for t in trajectory:
            center_tuple = tuple(t["center"])
            trajectory_map[center_tuple] = t.get("label", "")
        
        # Ekstrahuj labels v poradi expert_pairs
        # Kazdy expert_pair definuje transition state->action
        # Zacniname z prveho stavu a pokracujeme s kazdym action ako novy state
        labels = []
        processed_centers = set()
        
        for pair in expert_pairs:
            state = pair.get("state", [])
            action = pair.get("action", [])
            
            state_tuple = tuple(state)
            action_tuple = tuple(action)
            
            # Pridaj state label ak sme ho este nevideli
            if state_tuple in trajectory_map and state_tuple not in processed_centers:
                labels.append(trajectory_map[state_tuple])
                processed_centers.add(state_tuple)
            
            # Pridaj action label ak sme ho este nevideli
            if action_tuple in trajectory_map and action_tuple not in processed_centers:
                labels.append(trajectory_map[action_tuple])
                processed_centers.add(action_tuple)
        
        # Ulozi string cislic do textoveho suboru
        string_content = "".join(labels)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(string_content)
        
        print(f"[OK] {file_path} ({len(labels)} cislic: {string_content[:100]})")
    
    print(f"\n[HOTOVO] Stringy ulozene do priecinka '{output_dir}'")



if __name__ == "__main__":
    generate_string_files()

