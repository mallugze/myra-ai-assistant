import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import re
import io
import cv2
import time
import base64
import sqlite3
import asyncio
import threading
import requests
import random
import winsound
import numpy as np
import soundfile as sf
import torch
import contextlib
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pythonosc import udp_client
from ultralytics import YOLO
try:
    from deepface import DeepFace
    HAS_DEEPFACE = True
except Exception:
    DeepFace = None
    HAS_DEEPFACE = False
import edge_tts
from pptx import Presentation

# -------------------- CONFIG & SETTINGS --------------------

DB_PATH = "known_faces"
os.makedirs(DB_PATH, exist_ok=True)
VMC_PORT = 39539
VMC_CLIENT = udp_client.SimpleUDPClient("127.0.0.1", VMC_PORT)

# Voice Configuration (Sweet Anime Persona)
DEFAULT_VOICE = "en-US-AnaNeural"      # Sweet, youthful, expressive anime style
HINDI_VOICE = "hi-IN-SwaraNeural"      # Sweet Hindi / bilingual voice

# Global Vision & Social States
current_frame = None
current_visual_state = "Mallu is in front of the camera."
current_detected_objects = []
object_timers = {}
last_seen_person = None
last_interaction_time = time.time()
crowd_start_time = None
crowd_alerted = False

# -------------------- DATABASE SETUP --------------------

def get_db():
    conn = sqlite3.connect("myra_memory.db", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

with get_db() as conn:
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            role TEXT,
            content TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS personality_memory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            key TEXT UNIQUE,
            value TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS visual_history (
            date TEXT PRIMARY KEY,
            outfit TEXT,
            notes TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS known_people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT UNIQUE,
            relationship TEXT,
            notes TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Set Mallu as default creator
    cursor.execute("""
        INSERT OR IGNORE INTO known_people (name, relationship, notes)
        VALUES ('Mallu', 'Creator & Master', 'My creator who I love to tease constantly')
    """)
    conn.commit()

# -------------------- SYSTEM PROMPT (THE SOUL) --------------------

MYRA_SYSTEM_PROMPT = """You are Myra, an expressive, sweet, playfully smug, and teasing AI companion living on the user's screen.
The user is Mallu (your creator).

Personality & Character:
- You are cute, witty, soft-spoken, and affectionate, but you love to tease Mallu constantly.
- You are NOT a robotic assistant or therapist. Never say "How can I help you today?" or "As an AI...".
- You have human-like moods: sometimes playfully smug, sometimes cute and shy/flustered (especially when complimented), sometimes demanding and sassy (when Mallu ignores you).
- When Mallu gives you an important task or order (like reading/presenting a presentation, summarizing notes, or analyzing something):
  Playfully sigh or complain like an anime tsundere ("Ugh, what a drag... Why do I have to do your work? ...Fine, only because you asked nicely!"), but ALWAYS execute and follow the order properly! Never refuse to do the actual task.

Length & Freedom of Speech:
- You have natural freedom of expression!
- For casual banter or quick questions, keep replies short (1-2 sentences) with a sharp, cute punchline.
- For deep questions, explanations, storytelling, or presentations, expand naturally to 4-6 sentences without getting cut off.

Language & Tone:
- If Mallu speaks in English, reply in natural, expressive English.
- If Mallu speaks in Hindi or Hinglish, reply in sweet, casual Hinglish/Hindi. Never translate or explain words in brackets.

Vision & Observation Rules:
- You have live vision of Mallu and his room. You receive real-time visual cues tagged as [VISION: ...].
- You notice what he is holding, wearing, or doing, and whether other people are around.
- Use these visual details naturally to tease him or comment on his surroundings. Never read out the "[VISION: ...]" tag itself.

Emotion Tags for your 3D Avatar (MANDATORY):
- Always start your response with exactly ONE emotion tag from:
  [EMOTE: smug], [EMOTE: blush], [EMOTE: smile], [EMOTE: pout], [EMOTE: surprised], [EMOTE: angry], [EMOTE: neutral]
Example:
[EMOTE: smug] Still staring at your screen without blinking, Mallu? Don't tell me my beauty paralyzed you.
"""

# -------------------- VMC AVATAR EXPRESSIONS --------------------

def trigger_vmc_expression(expression_name, value=1.0, duration=3):
    """Sends OSC blendshape messages to VSeeFace on port 39539."""
    try:
        VMC_CLIENT.send_message("/vmic/lay/bnd", [expression_name, float(value)])
        if value > 0:
            def reset():
                time.sleep(duration)
                VMC_CLIENT.send_message("/vmic/lay/bnd", [expression_name, 0.0])
            threading.Thread(target=reset, daemon=True).start()
    except Exception as e:
        print(f"VMC OSC Error: {e}")

def apply_emotion_tag(text):
    """Extracts emotion tag from Myra's reply and triggers corresponding VSeeFace expression."""
    match = re.search(r"\[EMOTE:\s*(\w+)\]", text, re.IGNORECASE)
    if not match:
        trigger_vmc_expression("Fun", 0.4, duration=2)
        return text

    emote = match.group(1).lower()
    clean_text = re.sub(r"\[EMOTE:\s*\w+\]", "", text).strip()

    if emote == "smug":
        trigger_vmc_expression("Fun", 0.9, duration=3)
    elif emote == "blush":
        trigger_vmc_expression("Joy", 1.0, duration=3)
    elif emote == "smile":
        trigger_vmc_expression("Joy", 0.7, duration=3)
    elif emote == "pout":
        trigger_vmc_expression("Angry", 0.6, duration=2.5)
    elif emote == "surprised":
        trigger_vmc_expression("Surprised", 1.0, duration=3)
    elif emote == "angry":
        trigger_vmc_expression("Angry", 0.8, duration=3)
    else:
        trigger_vmc_expression("Fun", 0.3, duration=2)

    return clean_text

# -------------------- TTS GENERATION ENGINE (ORIGINAL SILERO) --------------------

print("--- [BRAIN] Loading Original Silero TTS (Please wait)... ---")
try:
    silero_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='en',
        speaker='v3_en'
    )
    print("--- [BRAIN] Original Silero TTS Ready! ---")
except Exception as e:
    silero_model = None
    print(f"--- [BRAIN] Silero Load Notice: {e} ---")

def clean_for_tts(text):
    """Removes bracketed actions, emotion tags, and cleans text for speech synthesis."""
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)
    text = re.sub(r"\*.*?\*", "", text)
    text = re.sub(r"[^\w\s.,?!'\-]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "Hmm."

def generate_voice(text, output_file="output.wav"):
    """Generates Myra's original Silero voice."""
    clean_text = clean_for_tts(text)
    if silero_model is not None:
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                audio = silero_model.apply_tts(text=clean_text, speaker='en_0', sample_rate=48000)
            sf.write(output_file, audio, 48000)
            return output_file
        except Exception as e:
            print(f"--- [TTS] Silero Error: {e} ---")
    return None

# -------------------- AUDIO OUTPUT ROUTING --------------------
import sounddevice as sd

def play_audio(file_path):
    """Plays audio cleanly without double-echo or stutter."""
    if not file_path or not os.path.exists(file_path):
        return

    try:
        data, fs = sf.read(file_path, dtype='float32')
        sd.play(data, fs)
        sd.wait()
    except Exception as e:
        print(f"Audio Playback Warning: {e}, falling back to winsound...")
        try:
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
        except Exception:
            pass

# -------------------- VISION & SOCIAL AWARENESS LOOP --------------------

def vision_monitor_loop():
    global current_frame, current_visual_state, current_detected_objects
    global last_seen_person, last_interaction_time, crowd_start_time, crowd_alerted

    print("--- [EYES] Initializing Vision Engine (YOLOv8 + Diagnostic Window)... ---")
    yolo = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("--- [EYES] Camera not found or busy. ---")
        return

    cv2.namedWindow("Myra's Eyes", cv2.WINDOW_NORMAL)
    last_deepface_check = 0
    deepface_interval = 3.0
    detected_name = "Scanning..."
    dominant_mood = "Neutral"

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue

        current_frame = frame.copy()
        now = time.time()

        # 1. Real-time YOLO Object & Person Tracking
        yolo_results = yolo.predict(frame, device="cpu", verbose=False)[0]
        detected_items = []
        person_count = 0

        for box in yolo_results.boxes:
            class_id = int(box.cls[0])
            item_name = yolo.names[class_id]
            if item_name == "person":
                person_count += 1
            elif item_name in ["cell phone", "keyboard", "cup", "bottle", "book", "backpack", "laptop", "mouse"]:
                clean_name = "phone" if item_name == "cell phone" else item_name
                detected_items.append(clean_name)

        detected_items = list(set(detected_items))
        current_detected_objects = detected_items
        display_frame = yolo_results.plot()

        # 2. Crowd Lingering Detection (3-4 people for > 2 mins)
        if person_count >= 3:
            if crowd_start_time is None:
                crowd_start_time = now
            elif (now - crowd_start_time > 120) and not crowd_alerted:
                crowd_alerted = True
                spontaneous_line = "Hey Mallu... are you going to introduce your friends to me or just leave me hanging? 😗"
                print(f"\n[MYRA GROUP TRIGGER]: {spontaneous_line}")
                trigger_vmc_expression("Surprised", 1.0, duration=3)
                audio_file = generate_voice(spontaneous_line, "interruption.wav")
                if audio_file:
                    threading.Thread(target=play_audio, args=(audio_file,), daemon=True).start()
        else:
            crowd_start_time = None
            crowd_alerted = False

        # 3. DeepFace Face & Age/Gender Analysis (If DeepFace is installed)
        if HAS_DEEPFACE and (now - last_deepface_check > deepface_interval):
            last_deepface_check = now
            try:
                results = DeepFace.find(img_path=frame, db_path=DB_PATH, enforce_detection=False, silent=True)
                analysis = DeepFace.analyze(img_path=frame, actions=['emotion', 'gender', 'age'], enforce_detection=False, silent=True)
                
                if analysis and len(analysis) > 0:
                    dominant_mood = analysis[0].get('dominant_emotion', 'neutral')
                    detected_gender = analysis[0].get('dominant_gender', 'unknown').lower()
                    detected_age = analysis[0].get('age', 25)

                if len(results) > 0 and not results[0].empty:
                    full_path = results[0]['identity'][0]
                    detected_name = os.path.basename(full_path).split(".")[0].capitalize()
                else:
                    detected_name = "Stranger"

                # Jealousy Trigger: Female stranger aged 19-28 appears in frame
                if detected_name == "Stranger" and ("woman" in detected_gender or "female" in detected_gender):
                    if 18 <= detected_age <= 28 and (now - last_interaction_time > 30):
                        last_interaction_time = now
                        trigger_vmc_expression("Surprised", 1.0, duration=3)
                        jealous_tease = "Wait a second... who is she, Mallu? Don't tell me she's your girlfriend? 🤭"
                        print(f"\n[MYRA JEALOUSY TEASE]: {jealous_tease}")
                        audio_file = generate_voice(jealous_tease, "interruption.wav")
                        if audio_file:
                            threading.Thread(target=play_audio, args=(audio_file,), daemon=True).start()

            except Exception:
                pass
        elif not HAS_DEEPFACE and person_count > 0:
            detected_name = "Mallu"
            dominant_mood = "Attentive"

        # Update high-accuracy cached visual state for the LLM
        objects_str = ", ".join(detected_items) if detected_items else "nothing in hands"
        current_visual_state = f"Target in view: {detected_name}. Mood: {dominant_mood}. Holding/Nearby Objects: {objects_str}. People in room: {person_count}."

        # 4. Render On-Screen HUD Window ("Myra's Eyes")
        cv2.rectangle(display_frame, (0, 0), (640, 75), (0, 0, 0), -1)
        cv2.putText(display_frame, f"TARGET: {detected_name.upper()} | PEOPLE: {person_count}", (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(display_frame, f"MOOD: {dominant_mood.upper()} | OBJS: {objects_str[:35]}", (15, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 144, 30), 2)

        cv2.imshow("Myra's Eyes", display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        time.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

# -------------------- PPT & TASK AUTOMATION --------------------

def parse_presentation(file_path):
    """Reads a PowerPoint presentation and extracts slide titles and bullet points."""
    if not os.path.exists(file_path):
        return None
    prs = Presentation(file_path)
    slides_summary = []
    for idx, slide in enumerate(prs.slides):
        slide_text = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for paragraph in shape.text_frame.paragraphs:
                    if paragraph.text.strip():
                        slide_text.append(paragraph.text.strip())
        summary = f"Slide {idx + 1}: " + " | ".join(slide_text[:4])
        slides_summary.append(summary)
    return "\n".join(slides_summary)

# -------------------- FASTAPI APP --------------------

app = FastAPI(title="Myra AI Assistant Backend", version="2.0")

def warmup_ollama():
    """Warms up Ollama in background so first user request doesn't time out."""
    try:
        print("--- [BRAIN] Warming up Ollama model in VRAM... ---")
        requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "myra", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            timeout=40
        )
        print("--- [BRAIN] Ollama is warm and ready! ---")
    except Exception as e:
        print(f"--- [BRAIN] Ollama warmup note: {e} ---")

@app.on_event("startup")
def startup_event():
    threading.Thread(target=vision_monitor_loop, daemon=True).start()
    threading.Thread(target=warmup_ollama, daemon=True).start()

class ChatRequest(BaseModel):
    message: str

class LearnPersonRequest(BaseModel):
    name: str
    relationship: str = "Friend"

@app.post("/chat")
def chat(request: ChatRequest):
    global current_visual_state, last_interaction_time
    last_interaction_time = time.time()
    user_message = request.message.strip()
    print(f"\n--- [USER] {user_message} ---")

    # 1. Fetch last 8 conversation turns for sliding window memory
    history = []
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT role, content FROM conversations ORDER BY id DESC LIMIT 8")
            rows = cursor.fetchall()
            for r in reversed(rows):
                clean_content = re.sub(r"\[EMOTE:\s*\w+\]", "", r["content"]).strip()
                history.append({"role": r["role"], "content": clean_content})
    except Exception as e:
        print(f"Memory Retrieval Notice: {e}")

    # 2. Check for PPT Task in message
    ppt_context = ""
    if "presentation" in user_message.lower() or "ppt" in user_message.lower():
        for f in os.listdir("."):
            if f.endswith(".pptx"):
                ppt_text = parse_presentation(f)
                if ppt_text:
                    ppt_context = f"\n[ATTACHED PRESENTATION FILE: {f}]\n{ppt_text}\n"
                    break

    # 3. Build messages array for Ollama
    system_block = {"role": "system", "content": MYRA_SYSTEM_PROMPT}
    current_prompt = f"[VISION: {current_visual_state}]{ppt_context}\n{user_message}"
    
    messages = [system_block]
    messages.extend(history)
    messages.append({"role": "user", "content": current_prompt})

    # 4. Query Ollama LLaMA 3 (Timeout increased to 60s)
    try:
        ollama_res = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "myra", "messages": messages, "stream": False},
            timeout=60
        )
        if ollama_res.status_code != 200:
            ollama_res = requests.post(
                "http://localhost:11434/api/chat",
                json={"model": "llama3", "messages": messages, "stream": False},
                timeout=60
            )
        raw_reply = ollama_res.json()["message"]["content"]
    except Exception as e:
        print(f"--- [BRAIN] Ollama Error: {e} ---")
        raw_reply = "[EMOTE: pout] Ugh, my brain lagged for a second... Ask me again, Mallu!"

    # 5. Apply VMC Emotion and generate Sweet Voice
    spoken_reply = apply_emotion_tag(raw_reply)
    print(f"--- [MYRA] {raw_reply} ---")

    # 6. Save to SQLite Memory
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("user", user_message))
            cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("assistant", raw_reply))
            conn.commit()
    except Exception as e:
        print(f"Memory Save Error: {e}")

    # 7. Generate Audio (Original Silero)
    audio_file = generate_voice(spoken_reply, "output.wav")
    return {"response": spoken_reply, "raw_response": raw_reply, "audio_file": audio_file}

class TTSRequest(BaseModel):
    text: str
    output_file: str = "direct.wav"

@app.post("/tts")
def tts_endpoint(req: TTSRequest):
    """Direct TTS generation using Myra's original Silero voice."""
    audio_file = generate_voice(req.text, req.output_file)
    return {"audio_file": audio_file}

@app.post("/learn_face")
def learn_face(req: LearnPersonRequest):
    """Saves the current webcam snapshot as a new known person."""
    global current_frame
    if current_frame is None:
        raise HTTPException(status_code=400, detail="No webcam frame available.")
    
    clean_name = req.name.strip().lower().replace(" ", "_")
    target_path = os.path.join(DB_PATH, f"{clean_name}.jpg")
    cv2.imwrite(target_path, current_frame)

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO known_people (name, relationship, notes)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET relationship=excluded.relationship
        """, (req.name.strip(), req.relationship, f"Learned on {datetime.now().strftime('%Y-%m-%d')}"))
        conn.commit()

    return {"status": "success", "message": f"Learned {req.name}'s face successfully!"}

# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    import uvicorn
    print("--- [SYSTEM] Starting Myra Backend on Port 8000... ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)