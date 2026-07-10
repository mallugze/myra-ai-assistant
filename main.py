import os
import re
import io
import cv2
import time
import base64
import sqlite3
import torch
import contextlib
import threading
import requests
import soundfile as sf
from fastapi import FastAPI
from pydantic import BaseModel
from deepface import DeepFace
from pythonosc import udp_client
from ultralytics import YOLO
from datetime import datetime, timedelta
import winsound
import random


# -------------------- CONFIG & SETTINGS --------------------

DB_PATH = "known_faces" 
VMC_PORT = 39539
VMC_CLIENT = udp_client.SimpleUDPClient("127.0.0.1", VMC_PORT)

# Global states
last_seen_person = None
last_interaction_time = 0
current_frame = None  
# Load the lightning-fast YOLO Nano model globally
yolo_model = YOLO("yolov8n.pt")
current_detected_objects = []



# -------------------- DATABASE & GLOBAL TRACKERS --------------------

# Active dictionary tracking how long objects are on your desk
# Format: { "phone": {"first_seen": timestamp, "last_seen": timestamp} }
object_timers = {}

conn = sqlite3.connect("myra_memory.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS conversations (id INTEGER PRIMARY KEY AUTOINCREMENT, role TEXT, content TEXT)")
cursor.execute("CREATE TABLE IF NOT EXISTS personality_memory (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE, value TEXT)")

# NEW: Persistent visual memory table for saving daily outfits
cursor.execute("CREATE TABLE IF NOT EXISTS visual_history (date TEXT PRIMARY KEY, outfit TEXT)")
conn.commit()

# -------------------- VMC ANIMATIONS --------------------

def trigger_vmc_expression(expression_name, value=1.0, duration=3):
    try:
        VMC_CLIENT.send_message("/vmic/lay/bnd", [expression_name, float(value)])
        if value > 0:
            def reset():
                time.sleep(duration)
                VMC_CLIENT.send_message("/vmic/lay/bnd", [expression_name, 0.0])
            threading.Thread(target=reset, daemon=True).start()
    except Exception as e:
        print(f"VMC Error: {e}")



# -------------------- VISION MONITOR LOOP --------------------

from ultralytics import YOLO
import os
import time
import cv2

# Force YOLO to run its core detection operations on the CPU
yolo_model = YOLO("yolov8n.pt")

# Global variable so Myra's chat endpoint can see detected objects
current_detected_objects = []

def vision_monitor_loop():
    global last_seen_person, last_interaction_time, current_frame, current_detected_objects
    
    print("--- [EYES] Initializing Live DL View Window (CPU Safe Mode)... ---")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("--- [EYES] ERROR: Camera is blocked or not found! ---")
        return

    cv2.namedWindow("Myra's Eyes", cv2.WINDOW_NORMAL)
    
    last_face_check = 0
    face_interval = 4.0  # Check face details every 4 seconds to save system resources
    
    name = "Scanning..."
    emotion = "Analyzing..."

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        
        current_frame = frame.copy()
        now = time.time()

        # 1. RUN OBJECT DETECTION CLEANLY ON CPU
        # Passing device="cpu" bypasses the Blackwell kernel crash entirely!
        yolo_results = yolo_model.predict(frame, device="cpu", verbose=False)[0]
        
        detected_items = []
        for box in yolo_results.boxes:
            class_id = int(box.cls[0])
            item_name = yolo_model.names[class_id]
            detected_items.append(item_name)
        
        current_detected_objects = list(set(detected_items))

        # Grab visual frame with tracking boxes overlayed
        display_frame = yolo_results.plot()

        # 2. RUN FACE/EMOTION DETECTION EVERY 4 SECONDS
        if now - last_face_check > face_interval:
            last_face_check = now
            try:
                # DeepFace handles CPU tracking natively if its source image is a clean NumPy array
                results = DeepFace.find(img_path=frame, db_path=DB_PATH, enforce_detection=False, silent=True)
                analysis = DeepFace.analyze(img_path=frame, actions=['emotion'], enforce_detection=False, silent=True)
                emotion = analysis[0]['dominant_emotion']

                if len(results) > 0 and not results[0].empty:
                    full_path = results[0]['identity'][0]
                    name = os.path.basename(full_path).split(".")[0]
                else:
                    name = "Stranger"

                # Interruption Engine
                if name != last_seen_person and (now - last_interaction_time > 15):
                    last_seen_person = name
                    last_interaction_time = now
                    if name == "Stranger":
                        trigger_vmc_expression("Surprised", 1.0)
                    else:
                        trigger_vmc_expression("Fun", 1.0)
                        
            except Exception as e:
                print(f"--- [EYES] Face Sync Error: {e} ---")

        # 3. DRAW DATA INTERFACE OVERLAY
        cv2.rectangle(display_frame, (0, 0), (640, 95), (0, 0, 0), -1)
        cv2.putText(display_frame, f"TARGET: {name.upper()}", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, f"MOOD: {emotion.upper()}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 144, 30), 2)
        
        if current_detected_objects:
            cv2.putText(display_frame, f"SEEING: {', '.join(current_detected_objects)}", (20, frame.shape[0] - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Myra's Eyes", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# -------------------- APP & MODELS --------------------

app = FastAPI()
@app.on_event("startup")
def startup_event():
    print("--- [SYSTEM] Triggering Vision Thread... ---")
    t = threading.Thread(target=vision_monitor_loop, daemon=True)
    t.start()

print("--- [BRAIN] Loading TTS Models (Please wait)... ---")
model_en, _ = torch.hub.load(repo_or_dir='snakers4/silero-models', model='silero_tts', language='en', speaker='v3_en')
print("--- [BRAIN] TTS Ready! ---")

class ChatRequest(BaseModel):
    message: str

def clean_for_tts(text):
    text = re.sub(r"[\*\(\[].*?[\*\)\]]", "", text)
    text = re.sub(r"[^\w\s.,?!]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "Hmm."

@app.post("/chat")
def chat(request: ChatRequest):
    global current_frame
    try:
        user_message = request.message.strip()
        print(f"--- [USER] {user_message} ---")

        visual_context = "The user is in front of you."
        if current_frame is not None:
            _, buffer = cv2.imencode('.jpg', current_frame)
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            try:
                vision_response = requests.post(
                    "http://localhost:11434/api/chat",
                    json={
                        "model": "moondream", 
                        "messages": [{"role": "user", "content": "Describe this scene in detail, including what the person is doing, what they are wearing, what objects they are holding, and any distinct items or background details visible.", "images": [image_b64]}],
                        "stream": False
                    }, timeout=5
                )
                visual_context = vision_response.json()["message"]["content"]
                print(f"--- [VISION] {visual_context} ---")
            except:
                print("--- [VISION] Moondream failed or is not running. ---")

        enriched_prompt = f"[OBSERVATION: {visual_context}]\nUser: {user_message}"
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "myra", "messages": [{"role": "user", "content": enriched_prompt}], "stream": False}
        )
        reply = response.json()["message"]["content"]
        reply_lower = reply.lower()

        # Expression logic
        if "?" in reply_lower:
            trigger_vmc_expression("Fun", 0.6, duration=4)
        elif any(word in reply_lower for word in ["haha", "funny", "joke", "tease"]):
            trigger_vmc_expression("Fun", 1.0, duration=3)
        elif any(word in reply_lower for word in ["no", "stop", "boring"]):
            trigger_vmc_expression("Angry", 0.5, duration=2)
        else:
            trigger_vmc_expression("Fun", 0.3, duration=2)

        cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("user", user_message))
        cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("assistant", reply))
        conn.commit()

        tts_text = clean_for_tts(reply)
        with contextlib.redirect_stdout(io.StringIO()):
            audio = model_en.apply_tts(text=tts_text, speaker='en_0', sample_rate=48000)

        filename = "output.wav"
        sf.write(filename, audio, 48000)
        return {"response": reply, "audio_file": filename}

    except Exception as e:
        print(f"--- [ERROR] Chat failed: {e} ---")
        return {"response": "My circuits are fried.", "audio_file": None}

# -------------------- STARTUP --------------------

if __name__ == "__main__":
    import uvicorn
    # 2. Start FastAPI Server
    print("--- [SYSTEM] Starting Server on Port 8000... ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)