import os
import io
import json
import base64
import pickle
import numpy as np
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_file, Response
import cv2
import face_recognition

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
PROFILES_DIR = os.path.join(BASE_DIR, "static", "profiles")
LOGS_DIR = os.path.join(BASE_DIR, "static", "logs")
MODEL_FILE = os.path.join(BASE_DIR, "trained_model.pkl")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(PROFILES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

STUDENTS_FILE = os.path.join(DATA_DIR, "students.json")
ATTENDANCE_FILE = os.path.join(DATA_DIR, "attendance.json")

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static")
)

# Global in-memory cache for trained model
trained_model_cache = None

def load_trained_model():
    global trained_model_cache
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                trained_model_cache = pickle.load(f)
                print(f"[Model loaded] Encodings: {len(trained_model_cache.get('encodings', []))}")
                return trained_model_cache
        except Exception as e:
            print(f"[Error loading model]: {e}")
    trained_model_cache = None
    return None

# Initial model load
load_trained_model()

def read_json_file(file_path, default_val):
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception:
            return default_val
    return default_val

def write_json_file(file_path, data):
    with open(file_path, "w") as f:
        json.dump(data, f, indent=4)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/stats", methods=["GET"])
def get_stats():
    students = read_json_file(STUDENTS_FILE, [])
    attendance = read_json_file(ATTENDANCE_FILE, [])
    
    today_str = datetime.now().strftime("%Y-%m-%d")
    today_records = [a for a in attendance if a.get("date") == today_str]
    
    present_student_ids = set(a["student_id"] for a in today_records if a.get("status") in ["Present", "Late"])
    
    total_students = len(students)
    present_today = len(present_student_ids)
    absent_today = max(0, total_students - present_today)
    attendance_rate = round((present_today / total_students * 100), 1) if total_students > 0 else 0
    
    # Department breakdown
    dept_counts = {}
    for s in students:
        d = s.get("department", "General")
        dept_counts[d] = dept_counts.get(d, 0) + 1
        
    model = load_trained_model()
    model_status = {
        "is_trained": model is not None and len(model.get("encodings", [])) > 0,
        "total_encodings": len(model.get("encodings", [])) if model else 0,
        "trained_at": model.get("trained_at", "N/A") if model else "N/A"
    }

    return jsonify({
        "status": "success",
        "total_students": total_students,
        "present_today": present_today,
        "absent_today": absent_today,
        "attendance_rate": attendance_rate,
        "model_status": model_status,
        "department_counts": dept_counts,
        "today_date": today_str
    })

@app.route("/api/students", methods=["GET", "POST"])
def manage_students():
    if request.method == "GET":
        students = read_json_file(STUDENTS_FILE, [])
        return jsonify({"status": "success", "students": students})
        
    elif request.method == "POST":
        try:
            data = request.get_json() or {}
            student_id = data.get("student_id") or f"STU{int(datetime.now().timestamp())}"
            name = data.get("name", "").strip()
            roll_no = data.get("roll_no", "").strip()
            department = data.get("department", "General").strip()
            email = data.get("email", "").strip()
            image_b64 = data.get("image")

            if not name or not roll_no:
                return jsonify({"status": "error", "message": "Name and Roll Number are required"}), 400

            # Save student entry
            students = read_json_file(STUDENTS_FILE, [])
            
            # Check duplicate roll number or student_id
            for s in students:
                if s["roll_no"] == roll_no or s["student_id"] == student_id:
                    return jsonify({"status": "error", "message": f"Student with Roll No '{roll_no}' already exists!"}), 400

            profile_pic_rel = f"/static/profiles/{student_id}.jpg"
            student_dir = os.path.join(DATASET_DIR, f"{student_id}_{name.replace(' ', '_')}")
            os.makedirs(student_dir, exist_ok=True)

            if image_b64:
                # Decode Base64 image
                if "," in image_b64:
                    image_b64 = image_b64.split(",")[1]
                img_bytes = base64.b64decode(image_b64)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if img is not None:
                    # Save image to dataset folder & profiles folder
                    face_save_path = os.path.join(student_dir, "face_1.jpg")
                    cv2.imwrite(face_save_path, img)
                    profile_save_path = os.path.join(PROFILES_DIR, f"{student_id}.jpg")
                    cv2.imwrite(profile_save_path, img)

            new_student = {
                "student_id": student_id,
                "roll_no": roll_no,
                "name": name,
                "department": department,
                "email": email,
                "profile_pic": profile_pic_rel,
                "registered_at": datetime.now().strftime("%Y-%m-%d")
            }
            students.append(new_student)
            write_json_file(STUDENTS_FILE, students)

            # Trigger retrain in background script
            from train_model import train as retrain_model
            retrain_model()
            load_trained_model()

            return jsonify({
                "status": "success",
                "message": f"Student {name} registered & model retrained successfully!",
                "student": new_student
            })

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/attendance", methods=["GET"])
def get_attendance():
    date_filter = request.args.get("date", "today")
    status_filter = request.args.get("status", "all")
    search_query = request.args.get("query", "").strip().lower()

    attendance = read_json_file(ATTENDANCE_FILE, [])
    today_str = datetime.now().strftime("%Y-%m-%d")

    # Filter by date
    if date_filter == "today":
        filtered = [a for a in attendance if a.get("date") == today_str]
    elif date_filter != "all" and date_filter:
        filtered = [a for a in attendance if a.get("date") == date_filter]
    else:
        filtered = list(attendance)

    # Filter by status
    if status_filter != "all":
        filtered = [a for a in filtered if a.get("status") == status_filter]

    # Search filter
    if search_query:
        filtered = [
            a for a in filtered
            if search_query in a.get("name", "").lower() or search_query in a.get("roll_no", "").lower()
        ]

    # Sort latest first
    filtered.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return jsonify({"status": "success", "count": len(filtered), "attendance": filtered})

@app.route("/api/attendance/manual", methods=["POST"])
def manual_attendance():
    try:
        data = request.get_json() or {}
        student_id = data.get("student_id")
        status = data.get("status", "Present")
        date_str = data.get("date", datetime.now().strftime("%Y-%m-%d"))

        students = read_json_file(STUDENTS_FILE, [])
        student = next((s for s in students if s["student_id"] == student_id), None)

        if not student:
            return jsonify({"status": "error", "message": "Student not found"}), 404

        attendance = read_json_file(ATTENDANCE_FILE, [])

        # Check existing entry for date
        existing = next((a for a in attendance if a["student_id"] == student_id and a["date"] == date_str), None)

        now = datetime.now()
        time_str = now.strftime("%I:%M:%S %p")
        timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S")

        if existing:
            existing["status"] = status
            existing["time"] = time_str
            existing["timestamp"] = timestamp_str
            existing["confidence"] = 100.0
            existing["source"] = "Manual Override"
        else:
            new_record = {
                "id": f"ATT_{int(now.timestamp() * 1000)}",
                "student_id": student["student_id"],
                "roll_no": student["roll_no"],
                "name": student["name"],
                "department": student["department"],
                "date": date_str,
                "time": time_str,
                "timestamp": timestamp_str,
                "status": status,
                "confidence": 100.0,
                "source": "Manual Entry"
            }
            attendance.append(new_record)

        write_json_file(ATTENDANCE_FILE, attendance)

        return jsonify({"status": "success", "message": f"Attendance for {student['name']} marked as {status}"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/attendance/<att_id>", methods=["DELETE"])
def delete_attendance(att_id):
    try:
        attendance = read_json_file(ATTENDANCE_FILE, [])
        new_attendance = [a for a in attendance if a.get("id") != att_id]
        if len(new_attendance) == len(attendance):
            return jsonify({"status": "error", "message": "Record not found"}), 404
        write_json_file(ATTENDANCE_FILE, new_attendance)
        return jsonify({"status": "success", "message": "Attendance record deleted successfully"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/attendance/clear", methods=["POST"])
def clear_attendance():
    try:
        write_json_file(ATTENDANCE_FILE, [])
        return jsonify({"status": "success", "message": "All attendance records cleared"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/students/<student_id>", methods=["DELETE"])
def delete_student(student_id):
    try:
        students = read_json_file(STUDENTS_FILE, [])
        new_students = [s for s in students if s.get("student_id") != student_id]
        if len(new_students) == len(students):
            return jsonify({"status": "error", "message": "Student not found"}), 404
        write_json_file(STUDENTS_FILE, new_students)
        from train_model import train as retrain_model
        retrain_model()
        load_trained_model()
        return jsonify({"status": "success", "message": "Student deleted and model retrained"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/recognize", methods=["POST"])
def recognize_face():
    try:
        data = request.get_json() or {}
        image_b64 = data.get("image")

        if not image_b64:
            return jsonify({"status": "error", "message": "No image payload received"}), 400

        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]

        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img_bgr is None:
            return jsonify({"status": "error", "message": "Invalid image data"}), 400

        # Optimization: Scale down high-resolution inputs to speed up face detection
        orig_h, orig_w = img_bgr.shape[:2]
        max_dim = max(orig_h, orig_w)
        scale = 1.0
        if max_dim > 900:
            scale = 900.0 / max_dim
            img_processed = cv2.resize(img_bgr, (int(orig_w * scale), int(orig_h * scale)))
        else:
            img_processed = img_bgr

        img_rgb = cv2.cvtColor(img_processed, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(img_rgb)
        face_encodings = face_recognition.face_encodings(img_rgb, face_locations)

        model = load_trained_model()
        if not model or not model.get("encodings"):
            return jsonify({
                "status": "error",
                "message": "Face recognition model is not trained yet. Please train the model first."
            }), 400

        known_encodings = model["encodings"]
        known_metadata = model["metadata"]

        results = []
        today_str = datetime.now().strftime("%Y-%m-%d")
        now_time = datetime.now()

        attendance_records = read_json_file(ATTENDANCE_FILE, [])
        attendance_updated = False

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            
            # Rescale box back to original coordinates
            orig_box = [int(top / scale), int(right / scale), int(bottom / scale), int(left / scale)]

            if len(distances) > 0:
                best_match_idx = np.argmin(distances)
                min_distance = distances[best_match_idx]

                TOLERANCE = 0.50  # Strict distance threshold for high precision
                if min_distance < TOLERANCE:
                    student_info = known_metadata[best_match_idx]
                    confidence = float(round(max(0, (1.0 - min_distance) * 100), 1))

                    student_id = student_info["student_id"]
                    
                    # Check if already marked today
                    existing_entry = next((a for a in attendance_records if a["student_id"] == student_id and a["date"] == today_str), None)

                    if not existing_entry:
                        # Determine status based on time (e.g. late if after 9:30 AM)
                        status = "Present"
                        if now_time.hour > 9 or (now_time.hour == 9 and now_time.minute > 30):
                            status = "Late"

                        new_record = {
                            "id": f"ATT_{int(now_time.timestamp() * 1000)}",
                            "student_id": student_id,
                            "roll_no": student_info["roll_no"],
                            "name": student_info["name"],
                            "department": student_info["department"],
                            "date": today_str,
                            "time": now_time.strftime("%I:%M:%S %p"),
                            "timestamp": now_time.strftime("%Y-%m-%d %H:%M:%S"),
                            "status": status,
                            "confidence": confidence,
                            "source": "Face AI Recognition"
                        }
                        attendance_records.append(new_record)
                        attendance_updated = True

                        action_status = "MARKED_PRESENT" if status == "Present" else "MARKED_LATE"
                        marked_time = now_time.strftime("%I:%M:%S %p")
                    else:
                        action_status = "ALREADY_MARKED"
                        marked_time = existing_entry.get("time", now_time.strftime("%I:%M:%S %p"))

                    results.append({
                        "recognized": True,
                        "box": orig_box,
                        "student": student_info,
                        "confidence": confidence,
                        "distance": round(float(min_distance), 4),
                        "action": action_status,
                        "marked_time": marked_time
                    })
                else:
                    results.append({
                        "recognized": False,
                        "box": orig_box,
                        "confidence": float(round(max(0, (1.0 - min_distance) * 100), 1)),
                        "action": "UNKNOWN_FACE"
                    })

        if attendance_updated:
            write_json_file(ATTENDANCE_FILE, attendance_records)

        return jsonify({
            "status": "success",
            "faces_detected": len(face_locations),
            "results": results
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/train", methods=["POST"])
def trigger_training():
    try:
        from train_model import train as run_training
        success = run_training()
        load_trained_model()
        if success:
            return jsonify({"status": "success", "message": "Model trained successfully!"})
        else:
            return jsonify({"status": "error", "message": "Model training failed."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/api/export", methods=["GET"])
def export_attendance_csv():
    attendance = read_json_file(ATTENDANCE_FILE, [])
    
    csv_buffer = io.StringIO()
    csv_buffer.write("ID,Student ID,Roll No,Name,Department,Date,Time,Status,Confidence (%),Source\n")
    
    for a in attendance:
        line = f'"{a.get("id")}","{a.get("student_id")}","{a.get("roll_no")}","{a.get("name")}","{a.get("department")}","{a.get("date")}","{a.get("time")}","{a.get("status")}","{a.get("confidence")}","{a.get("source")}"\n'
        csv_buffer.write(line)
        
    mem = io.BytesIO()
    mem.write(csv_buffer.getvalue().encode('utf-8'))
    mem.seek(0)
    
    filename = f"attendance_records_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return send_file(mem, mimetype="text/csv", as_attachment=True, download_name=filename)

if __name__ == "__main__":
    print("Starting AI Attendance Portal Server on http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True, use_reloader=False)
