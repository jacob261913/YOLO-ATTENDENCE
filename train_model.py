import os
import pickle
import json
import numpy as np
import face_recognition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
MODEL_FILE = os.path.join(BASE_DIR, "trained_model.pkl")
STUDENTS_JSON = os.path.join(BASE_DIR, "data", "students.json")

def train():
    print("Starting face recognition model training...")
    
    known_encodings = []
    known_metadata = []
    
    # Load student metadata dictionary if available
    students_meta = {}
    if os.path.exists(STUDENTS_JSON):
        with open(STUDENTS_JSON, "r") as f:
            st_list = json.load(f)
            students_meta = {s["student_id"]: s for s in st_list}

    if not os.path.exists(DATASET_DIR):
        print("Dataset directory does not exist.")
        return False
        
    student_folders = [f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    
    total_images_processed = 0
    total_encodings_generated = 0

    for folder_name in student_folders:
        folder_path = os.path.join(DATASET_DIR, folder_name)
        # folder_name format: STU001_Alex_Johnson or just student_id
        parts = folder_name.split("_")
        student_id = parts[0]
        
        # Get metadata
        meta = students_meta.get(student_id, {
            "student_id": student_id,
            "name": " ".join(parts[1:]) if len(parts) > 1 else student_id,
            "roll_no": f"ROLL-{student_id}",
            "department": "General",
            "profile_pic": f"/static/profiles/{student_id}.jpg"
        })
        
        image_files = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
        
        for img_name in image_files:
            total_images_processed += 1
            img_path = os.path.join(folder_path, img_name)
            try:
                import cv2
                image_bgr = cv2.imread(img_path)
                if image_bgr is None:
                    continue
                # Resize if image is excessively large to optimize dlib face detector
                h, w = image_bgr.shape[:2]
                max_dim = max(h, w)
                if max_dim > 1200:
                    scale = 1200.0 / max_dim
                    image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
                
                image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
                encodings = face_recognition.face_encodings(image_rgb)
                
                if encodings:
                    for enc in encodings:
                        known_encodings.append(enc)
                        known_metadata.append(meta)
                        total_encodings_generated += 1
                    print(f"  [+] Encodings generated for {meta['name']} ({img_name})")
                else:
                    print(f"  [-] No face detected in {img_path}")
            except Exception as e:
                print(f"  [!] Error processing {img_path}: {e}")

    from datetime import datetime
    model_data = {
        "encodings": known_encodings,
        "metadata": known_metadata,
        "total_students": len(set(m["student_id"] for m in known_metadata)),
        "total_encodings": total_encodings_generated,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)

    print(f"\nTraining Complete!")
    print(f"Total Students Trained: {model_data['total_students']}")
    print(f"Total Images Processed: {total_images_processed}")
    print(f"Total Encodings Saved: {total_encodings_generated}")
    print(f"Model File Saved: {MODEL_FILE}")
    return True

if __name__ == "__main__":
    train()
