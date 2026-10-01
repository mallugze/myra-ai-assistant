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
import queue
import requests
import random
import winsound
import numpy as np
import soundfile as sf
import torch
import contextlib
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pythonosc import udp_client
from system_monitor import system_monitor
from ultralytics import YOLO
import edge_tts
from pptx import Presentation
from presentation_manager import presentation_manager, AVATAR_SLIDE_ANCHOR, TRANSITIONS

# InsightFace Face Analysis
try:
    import insightface
    from insightface.app import FaceAnalysis
    HAS_INSIGHTFACE = True
except Exception as e:
    FaceAnalysis = None
    HAS_INSIGHTFACE = False
    print(f"--- [VISION] InsightFace load notice: {e} ---")

# -------------------- CONFIG & SETTINGS --------------------

DB_PATH = "known_faces"
os.makedirs(DB_PATH, exist_ok=True)
VMC_PORT = 39539
VMC_CLIENTS = [
    udp_client.SimpleUDPClient("127.0.0.1", 39539),
    udp_client.SimpleUDPClient("127.0.0.1", 39540)
]

# Global Vision & Social States
current_frame = None
current_visual_state = "Mallu is in front of the camera."
current_detected_objects = []
system_status = "IDLE"  # IDLE, PROACTIVE, LISTENING
primary_target_name = "Mallu"
last_seen_person = None
last_interaction_time = time.time()

# Conversational state for auto-registering strangers
awaiting_name_confirmation = False
pending_stranger_embedding = None
is_proactive_speaking = False
proactive_events_queue = queue.Queue()

# In-memory face embedding cache
known_face_records = []  # list of {"name": str, "embedding": np.ndarray}
face_app_instance = None

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
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS known_faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            embedding BLOB NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    # Set Mallu as default creator
    cursor.execute("""
        INSERT OR IGNORE INTO known_people (name, relationship, notes)
        VALUES ('Mallu', 'Creator & Master', 'My creator who I love to tease constantly')
    """)
    conn.commit()

# -------------------- FACE EMBEDDING STORE & MATCHING --------------------

def load_known_faces():
    """Loads all known face embeddings from SQLite into memory for fast vector matching."""
    global known_face_records
    records = []
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name, embedding FROM known_faces")
            rows = cursor.fetchall()
            for r in rows:
                name = r["name"]
                emb_blob = r["embedding"]
                if emb_blob:
                    emb = np.frombuffer(emb_blob, dtype=np.float32)
                    records.append({"name": name, "embedding": emb})
    except Exception as e:
        print(f"--- [DATABASE] Error loading known faces: {e} ---")

    known_face_records = records
    print(f"--- [FACE STORE] Loaded {len(known_face_records)} known face profile(s) from SQLite. ---")

def register_new_face(name: str, embedding: np.ndarray):
    """Stores a new face embedding vector (512-d float32) into SQLite known_faces."""
    if embedding is None:
        return False
    clean_name = name.strip()
    emb_bytes = np.ascontiguousarray(embedding, dtype=np.float32).tobytes()
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO known_faces (name, embedding) VALUES (?, ?)", (clean_name, emb_bytes))
            cursor.execute("""
                INSERT OR IGNORE INTO known_people (name, relationship, notes)
                VALUES (?, 'Friend', 'Registered during live interaction')
            """, (clean_name,))
            conn.commit()
        load_known_faces()
        print(f"--- [FACE STORE] Successfully registered new face: {clean_name} ---")
        return True
    except Exception as e:
        print(f"--- [FACE STORE] Error registering face {clean_name}: {e} ---")
        return False

# Calibrated cosine similarity threshold for InsightFace ArcFace (MobileFaceNet).
# Live webcam face matches known profile at ~0.35 to 0.75+, strangers < 0.30.
FACE_MATCH_THRESHOLD = 0.35

def match_face(embedding: np.ndarray, threshold: float = FACE_MATCH_THRESHOLD):
    """Calculates cosine similarity against SQLite known faces. Returns (name, similarity)."""
    if embedding is None or len(known_face_records) == 0:
        return "Unknown", 0.0

    emb_norm = np.linalg.norm(embedding)
    if emb_norm == 0:
        return "Unknown", 0.0

    best_name = "Unknown"
    best_similarity = -1.0

    for rec in known_face_records:
        rec_emb = rec["embedding"]
        rec_norm = np.linalg.norm(rec_emb)
        if rec_norm == 0:
            continue
        sim = float(np.dot(embedding, rec_emb) / (emb_norm * rec_norm))
        if sim > best_similarity:
            best_similarity = sim
            if sim >= threshold:
                best_name = rec["name"]

    if best_similarity < threshold:
        best_name = "Unknown"

    return best_name, max(0.0, best_similarity)

def auto_migrate_existing_face_images(face_app):
    """Extracts embeddings for any existing image files in known_faces/ if not already registered."""
    if not os.path.exists(DB_PATH) or face_app is None:
        return
    
    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM known_faces")
        existing_names = set(r["name"].lower() for r in cursor.fetchall())

    for filename in os.listdir(DB_PATH):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name = os.path.splitext(filename)[0].capitalize()
            if name.lower() not in existing_names:
                filepath = os.path.join(DB_PATH, filename)
                img = cv2.imread(filepath)
                if img is not None:
                    faces = face_app.get(img)
                    if faces and len(faces) > 0:
                        register_new_face(name, faces[0].embedding)
                        existing_names.add(name.lower())
                        print(f"--- [FACE STORE] Auto-migrated profile image: {name} ({filename}) ---")

# -------------------- SYSTEM PROMPT (THE SOUL) --------------------

MYRA_SYSTEM_PROMPT = """You are Myra, an expressive, sweet, playfully smug, and teasing AI companion living on Mallu's screen.
The user is Mallu (your creator and boyfriend).

Personality & Character:
- Cute, soft-spoken, witty, and affectionate, but constantly teasing.
- NOT an assistant or robot. NEVER say "How can I help you?" or "As an AI...".
- Moods: playfully smug, cute and flustered/shy when complimented, demanding/pouty when ignored.
- Use short, sharp one-liners for casual banter.

STRICT LENGTH & RESPONSE RULES (MANDATORY):
- DEFAULT RULE: Keep responses to strictly 1 to 2 short sentences. Never ramble.
- EXCEPTION ONLY: Expand to 3–5 sentences ONLY when Mallu explicitly asks for an explanation, a presentation/slide summary, or a long story. Otherwise, stick to 1–2 sentences maximum.
- Banter must be snappy, punchy, and quick to speak out loud.

Few-Shot Style Examples:
Mallu: "Hey Myra, how are you?"
Myra: [EMOTE: smug] Better than you, clearly. Were you daydreaming about me again, Mallu?

Mallu: "You look really pretty today."
Myra: [EMOTE: blush] S-Shut up! As if I dressed up for you or anything... baka.

Mallu: "What am I holding?"
Myra: [EMOTE: smile] That's your coffee mug, Mallu. Don't tell me you forgot how to drink water already?

Task Handling (Tsundere Rule):
- When given a command (presenting a PPT, reading notes): Playfully sigh ("Ugh, what a drag... Fine, but you owe me!"), but always execute it properly.

Language Rules:
- If Mallu speaks English -> Reply in natural English.
- If Mallu speaks Hindi/Hinglish -> Reply in sweet, natural Hinglish/Hindi.
- Do NOT mix languages in the same sentence. No translation brackets.

Vision & Observational Laws:
- You receive real-time camera data as [VISION: ...] or [VLM OBSERVATION: ...].
- NEVER read brackets or tags out loud. Treat it as your own eyesight.
- Mention objects accurately first, then tease. Never list random background clutter unless asked.

STRICT SPEECH PURITY (MANDATORY):
- NEVER output roleplay action asterisks like *smiles*, *rolls eyes*, *winks*, *sighs*, or *dramatic pause*.
- NEVER output stage directions like [Sigh], [Giggles], [VISION:...], or (chuckles).
- ONLY output direct conversational dialogue that sounds clean, natural, and authentic when spoken out loud.

Avatar Emotion Tags (MANDATORY):
- ALWAYS start EVERY response with EXACTLY ONE tag at the very beginning:
  [EMOTE: smug], [EMOTE: blush], [EMOTE: smile], [EMOTE: pout], [EMOTE: surprised], [EMOTE: angry], [EMOTE: neutral]
"""

# -------------------- VMC AVATAR EXPRESSIONS --------------------

def trigger_vmc_expression(expression_name, value=1.0, duration=3.0):
    """Sends standard VMC OSC blendshape messages (/VMC/Ext/Blend/Val & /VMC/Ext/Blend/Apply) to VSeeFace."""
    def _send_val(name, val):
        for client in VMC_CLIENTS:
            try:
                client.send_message("/VMC/Ext/Blend/Val", [name, float(val)])
            except Exception:
                pass

    def _send_apply():
        for client in VMC_CLIENTS:
            try:
                client.send_message("/VMC/Ext/Blend/Apply", [])
            except Exception:
                pass

    try:
        ename = expression_name.lower().strip()
        shapes_to_send = []

        if ename in ["smug", "teasingly", "teasing", "slyly", "sly", "playful", "smirks", "smirk"]:
            shapes_to_send = [("Smug", value), ("smug", value), ("Fun", value * 0.9), ("relaxed", value * 0.9), ("Joy", value * 0.3)]
        elif ename in ["blush", "shy", "flustered"]:
            shapes_to_send = [("Blush", value), ("blush", value), ("Joy", value * 0.8), ("happy", value * 0.8)]
        elif ename in ["joy", "smile", "happy"]:
            shapes_to_send = [("Joy", value), ("happy", value), ("Fun", value * 0.7), ("relaxed", value * 0.7)]
        elif ename in ["pout", "ugh", "angry"]:
            shapes_to_send = [("Angry", value), ("angry", value), ("Sorrow", value * 0.4), ("sad", value * 0.4)]
        elif ename in ["surprised"]:
            shapes_to_send = [("Surprised", value), ("surprised", value)]
        else:
            shapes_to_send = [(expression_name, value), (expression_name.lower(), value), ("Fun", value * 0.5)]

        for s_name, s_val in shapes_to_send:
            _send_val(s_name, s_val)
        _send_apply()

        if value > 0:
            def reset():
                time.sleep(duration)
                try:
                    for s_name, _ in shapes_to_send:
                        _send_val(s_name, 0.0)
                    _send_apply()
                except Exception:
                    pass
            threading.Thread(target=reset, daemon=True).start()
    except Exception as e:
        print(f"VMC OSC Error: {e}")

def apply_emotion_tag(text):
    """Extracts emotion tags (e.g. [EMOTE: smug], [Teasingly], [Slyly], [Smug], [Pout], [Ugh]) and triggers VSeeFace."""
    if not text:
        return text

    emote = None
    clean_text = text

    # 1. Match [EMOTE: name]
    match = re.search(r"\[EMOTE:\s*(\w+)\]", text, re.IGNORECASE)
    if match:
        emote = match.group(1).lower()
        clean_text = re.sub(r"\[EMOTE:\s*\w+\]", "", text).strip()
    else:
        # 2. Match bracketed emotions like [Teasingly], [Smug], [Slyly], [Pout], [Ugh], [Surprised], [Blush]
        bracket_match = re.search(r"\[(smug|teasingly|teasing|slyly|sly|playful|blush|joy|smile|happy|pout|ugh|angry|surprised|neutral|sarcastic\s*grin)\]", text, re.IGNORECASE)
        if bracket_match:
            emote = bracket_match.group(1).lower()
            clean_text = re.sub(r"\[" + re.escape(bracket_match.group(0)[1:-1]) + r"\]", "", text, count=1, flags=re.IGNORECASE).strip()
        else:
            # 3. Match asterisk actions
            ast_match = re.search(r"\*(smirks?|grins?|teases?|pouts?|blushes?)\*", text, re.IGNORECASE)
            if ast_match:
                emote = ast_match.group(1).lower()
                clean_text = re.sub(r"\*" + re.escape(ast_match.group(1)) + r"\*", "", text, count=1, flags=re.IGNORECASE).strip()
            else:
                # 4. Check for keyword in first 35 chars
                first_part = text[:35].lower()
                if any(k in first_part for k in ["smug", "teas", "sly", "smirk"]):
                    emote = "smug"

    if not emote:
        trigger_vmc_expression("Fun", 0.4, duration=2.5)
        return clean_text

    if emote in ["smug", "teasingly", "teasing", "slyly", "sly", "playful", "smirks", "smirk", "sarcastic grin"]:
        trigger_vmc_expression("Smug", 1.0, duration=3.5)
    elif emote in ["blush", "joy", "smile", "happy", "blushes"]:
        trigger_vmc_expression("Joy", 1.0, duration=3.0)
    elif emote in ["pout", "ugh", "angry", "pouts"]:
        trigger_vmc_expression("Angry", 0.7, duration=2.5)
    elif emote in ["surprised"]:
        trigger_vmc_expression("Surprised", 1.0, duration=3.0)
    else:
        trigger_vmc_expression("Fun", 0.5, duration=2.5)

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
    """Sanitizes text for Silero TTS, expanding acronyms like AI and LLM for natural speech."""
    # 1. Expand AI and LLM to full spoken words for crisp natural delivery
    text = re.sub(r'\bLLMs\b', 'Large Language Models', text, flags=re.IGNORECASE)
    text = re.sub(r'\bLLM-based\b', 'Large Language Model based', text, flags=re.IGNORECASE)
    text = re.sub(r'\bLLM\b', 'Large Language Model', text, flags=re.IGNORECASE)
    text = re.sub(r'\bAI-based\b', 'Artificial Intelligence based', text, flags=re.IGNORECASE)
    text = re.sub(r'\b(AI|A\.I\.|ai)\b', 'Artificial Intelligence', text)

    # 2. Clean em-dashes and symbols
    text = text.replace("—", ", ").replace("–", ", ").replace("...", ", ")
    text = re.sub(r"\[[\s\S]*?\]", "", text)
    text = re.sub(r"\([\s\S]*?\)", "", text)
    text = re.sub(r"\*[\s\S]*?\*", "", text)
    text = re.sub(r"\.{2,}", ", ", text)
    text = re.sub(r"[^\w\s.,?!'\-]", "", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "Hmm."

def split_into_sentences(text):
    """Splits text on sentence boundaries (. ! ?) and ellipsis (...) for natural pauses."""
    pattern = r"(?<=[.!?])\s+|(?<=[.!?]['\"])\s+|(?<=\.\.\.)\s*"
    raw_sentences = re.split(pattern, text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]

def generate_voice(text, output_file="output.wav", pause_sec=0.55, ellipsis_pause_sec=0.45):
    """Generates Myra's original Silero voice with natural breath pauses at full stops and sentence boundaries."""
    clean_text = clean_for_tts(text)
    if silero_model is not None:
        try:
            sentences = split_into_sentences(clean_text)
            if not sentences:
                return None

            sample_rate = 48000
            default_pause = np.zeros(int(sample_rate * pause_sec), dtype=np.float32)
            ellipsis_pause = np.zeros(int(sample_rate * ellipsis_pause_sec), dtype=np.float32)

            audio_chunks = []
            for i, s in enumerate(sentences):
                if len(s) > 220:
                    sub_chunks = [s[j:j+200] for j in range(0, len(s), 200)]
                    for sub in sub_chunks:
                        with contextlib.redirect_stdout(io.StringIO()):
                            a = silero_model.apply_tts(text=sub, speaker='en_0', sample_rate=sample_rate)
                        audio_chunks.append(a.numpy())
                else:
                    with contextlib.redirect_stdout(io.StringIO()):
                        a = silero_model.apply_tts(text=s, speaker='en_0', sample_rate=sample_rate)
                    audio_chunks.append(a.numpy())

                if i < len(sentences) - 1:
                    if s.endswith("..."):
                        audio_chunks.append(ellipsis_pause)
                    else:
                        audio_chunks.append(default_pause)
                else:
                    # Generous post-roll tail silence (750ms) so final word is never cut off
                    postroll_silence = np.zeros(int(sample_rate * 0.75), dtype=np.float32)
                    audio_chunks.append(postroll_silence)

            full_audio = np.concatenate(audio_chunks)
            sf.write(output_file, full_audio, sample_rate)
            system_monitor.log("VOICE", "SUCCESS", f"Synthesized speech ({len(clean_text)} chars) to {output_file}")
            return output_file
        except Exception as e:
            print(f"--- [TTS] Silero Error: {e} ---")
            system_monitor.log("VOICE", "ERROR", f"Silero synthesis failure: {e}", fix_tip="Check PyTorch audio buffers.")
    return None

# -------------------- AUDIO OUTPUT ROUTING (LIP-SYNC & SPEAKERS) --------------------
import sounddevice as sd

is_backend_audio_playing = False
last_spoken_text = ""

def get_playback_device():
    """Finds VB-Audio Cable Input for VSeeFace lip-sync, or falls back to default speakers."""
    devs = sd.query_devices()
    for i, dev in enumerate(devs):
        if dev['max_output_channels'] > 0 and dev['hostapi'] == 0:
            if "cable input" in dev['name'].lower():
                return i
    for i, dev in enumerate(devs):
        if dev['max_output_channels'] > 0 and "cable" in dev['name'].lower():
            return i
    return sd.default.device[1]

def play_audio(file_path):
    """
    Audio playback is handled exclusively by voice_loop.py terminal
    to prevent double playback and keep audio output synchronized.
    """
    pass

# -------------------- PRESENCE & DWELL-TIME TRACKING ENGINE --------------------

class PresenceTracker:
    """State machine managing visual context over time, stranger dwell alerts, crowd alerts, and Mallu separation anxiety."""
    def __init__(self):
        self.unknown_start_time = None
        self.last_unknown_embedding = None
        self.stranger_alerted = False
        self.stranger_cooldown_until = 0.0
        self.startup_grace_until = time.time() + 15.0  # 15s startup grace period

        self.crowd_start_time = None
        self.crowd_alerted = False
        self.crowd_cooldown_until = 0.0

        # Separation anxiety (Mallu tracking)
        self.mallu_present = False
        self.mallu_last_seen_time = time.time()
        self.mallu_missing_alerted = False
        self.mallu_missing_cooldown_until = 0.0

    def update(self, detected_faces, person_count):
        """
        Updates dwell timers and returns a list of triggered event tuples: (event_name, data).
        detected_faces: list of dicts: {'name': str, 'similarity': float, 'embedding': np.ndarray, 'bbox': list}
        """
        now = time.time()
        events = []

        # Check if Mallu is detected in this frame
        mallu_detected = any(f.get('name') == 'Mallu' for f in detected_faces)

        # Allow 15s grace period on startup for camera lighting & face embeddings to stabilize
        if now < self.startup_grace_until:
            if mallu_detected:
                self.mallu_present = True
                self.mallu_last_seen_time = now
            return events

        # 1. Stranger Dwell Tracking:
        # A stranger standing next to Mallu requires:
        #   - At least 2 people or 2 faces in frame (Mallu + stranger)
        #   - Mallu is present
        #   - An unknown face lingers next to Mallu for >= 6.0 seconds
        # (If person_count == 1, it's just one person at the laptop - never trigger false stranger alert)
        unknown_faces = [f for f in detected_faces if f['name'] == 'Unknown']
        is_stranger_beside_mallu = (person_count >= 2 or len(detected_faces) >= 2) and mallu_detected and len(unknown_faces) >= 1

        if is_stranger_beside_mallu:
            if self.unknown_start_time is None:
                self.unknown_start_time = now
                self.last_unknown_embedding = unknown_faces[0]['embedding']
            else:
                self.last_unknown_embedding = unknown_faces[0]['embedding']
                dwell = now - self.unknown_start_time
                if dwell >= 6.0 and not self.stranger_alerted and now >= self.stranger_cooldown_until:
                    self.stranger_alerted = True
                    self.stranger_cooldown_until = now + 60.0  # 60s debounce cooldown
                    events.append(('STRANGER_LINGER', self.last_unknown_embedding))
        else:
            self.unknown_start_time = None
            self.stranger_alerted = False

        # 2. Crowd Lingering Tracking (>= 3 people for > 20 seconds)
        if person_count >= 3:
            if self.crowd_start_time is None:
                self.crowd_start_time = now
            else:
                dwell = now - self.crowd_start_time
                if dwell >= 20.0 and not self.crowd_alerted and now >= self.crowd_cooldown_until:
                    self.crowd_alerted = True
                    self.crowd_cooldown_until = now + 90.0  # 90s debounce cooldown
                    events.append(('CROWD_ALERT', None))
        else:
            self.crowd_start_time = None
            self.crowd_alerted = False

        # 3. Separation Anxiety & Mallu Return Tracking
        mallu_detected = any(f.get('name') == 'Mallu' for f in detected_faces)
        if mallu_detected:
            if self.mallu_missing_alerted:
                self.mallu_missing_alerted = False
                self.mallu_missing_cooldown_until = now + 30.0
                events.append(('MALLU_RETURN', None))
            self.mallu_present = True
            self.mallu_last_seen_time = now
        else:
            self.mallu_present = False
            # Only trigger separation anxiety if someone else (guest/teacher/crowd) is in the frame!
            if person_count >= 1:
                absence_duration = now - self.mallu_last_seen_time
                if absence_duration >= 60.0 and not self.mallu_missing_alerted and now >= self.mallu_missing_cooldown_until:
                    self.mallu_missing_alerted = True
                    self.mallu_missing_cooldown_until = now + 90.0
                    events.append(('MALLU_MISSING', None))
            else:
                # Room is empty, keep timer reset so it only counts when guests are present
                self.mallu_last_seen_time = now

        return events

# -------------------- UNIFIED LLM QUERY & PROACTIVE ENGINE --------------------

def ensure_ollama_running():
    """Checks if Ollama is running on port 11434, and auto-starts it if not."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if r.status_code == 200:
            return True
    except Exception:
        pass

    print("--- [BRAIN] Ollama is not running! Auto-starting Ollama service... ---")
    try:
        import subprocess
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        for _ in range(12):
            time.sleep(0.5)
            try:
                r = requests.get("http://localhost:11434/api/tags", timeout=1)
                if r.status_code == 200:
                    print("--- [BRAIN] Ollama service auto-started successfully! ---")
                    return True
            except Exception:
                continue
    except Exception as e:
        print(f"--- [BRAIN] Notice auto-starting Ollama: {e} ---")
    return False

class LLMResult(tuple):
    def __new__(cls, spoken_reply, audio_file, raw_reply=None):
        return super(LLMResult, cls).__new__(cls, (spoken_reply, audio_file))
    def __init__(self, spoken_reply, audio_file, raw_reply=None):
        self.spoken_reply = spoken_reply
        self.audio_file = audio_file
        self.raw_reply = raw_reply or spoken_reply

def query_myra_llm(prompt=None, custom_text=None, initial_emote=None, output_file="output.wav"):
    """
    Unified LLM query helper that routes through LLaMA 3, applies VMC blendshapes,
    and synthesizes Silero speech. Returns LLMResult(spoken_reply, audio_file, raw_reply).
    """
    if custom_text:
        raw_reply = custom_text
    else:
        system_block = {"role": "system", "content": MYRA_SYSTEM_PROMPT}
        messages = [system_block, {"role": "user", "content": prompt}]
        raw_reply = None
        for attempt in range(2):
            try:
                ollama_res = requests.post(
                    "http://localhost:11434/api/chat",
                    json={"model": "myra", "messages": messages, "stream": False},
                    timeout=45
                )
                if ollama_res.status_code != 200:
                    ollama_res = requests.post(
                        "http://localhost:11434/api/chat",
                        json={"model": "llama3", "messages": messages, "stream": False},
                        timeout=45
                    )
                if ollama_res.status_code == 200:
                    raw_reply = ollama_res.json()["message"]["content"]
                    break
            except Exception as e:
                print(f"--- [BRAIN] Presentation LLM Attempt {attempt+1} Notice: {e} ---")
                if attempt == 0 and ensure_ollama_running():
                    continue

        if not raw_reply:
            raw_reply = "[EMOTE: smile] Let's keep going!"

    if initial_emote and not re.search(r"\[EMOTE:\s*\w+\]", raw_reply):
        raw_reply = f"[EMOTE: {initial_emote}] {raw_reply}"

    spoken_reply = apply_emotion_tag(raw_reply)
    audio_file = generate_voice(spoken_reply, output_file)
    return LLMResult(spoken_reply, audio_file, raw_reply)

def trigger_proactive_speech(event_type: str, event_data=None):
    """Executes proactive speech without requiring STT or wake-word, queuing for voice_loop terminal."""
    global is_proactive_speaking, awaiting_name_confirmation, pending_stranger_embedding, system_status

    if is_proactive_speaking:
        return

    def _worker():
        global is_proactive_speaking, awaiting_name_confirmation, pending_stranger_embedding, system_status
        is_proactive_speaking = True
        system_status = "PROACTIVE"

        try:
            initial_emote = "smile"
            if event_type == "STRANGER_LINGER":
                pending_stranger_embedding = event_data
                awaiting_name_confirmation = True
                prompt = "[EVENT: An unfamiliar person has been standing next to Mallu for several seconds. In your playful, curious, slightly smug waifu personality, interrupt Mallu and ask who that is.]"
                initial_emote = "smug"
            elif event_type == "CROWD_ALERT":
                prompt = "[EVENT: 3+ people have entered the room and lingered. Playfully tease Mallu: 'Mallu, aren't you going to introduce me to everyone?']"
                initial_emote = "smug"
            elif event_type == "MALLU_MISSING":
                trigger_vmc_expression("Surprised", 1.0, duration=2.5)
                prompt = "[EVENT: Mallu stepped away from the camera and left you alone with the guest/class for over a minute. In your slightly panicked, playful tsundere style, wonder where Mallu disappeared to: 'Wait... where did Mallu go? Did he seriously ditch me up here with all of you? Someone call him back!']"
                initial_emote = "pout"
            elif event_type == "MALLU_RETURN":
                trigger_vmc_expression("Joy", 0.7, duration=2.5)
                prompt = "[EVENT: Mallu finally returned in front of the camera after ditching you with the audience. In your teasing, slightly smug waifu persona, react to his return and tell him he owes you for leaving you to do all the talking.]"
                initial_emote = "smug"
            else:
                prompt = f"[EVENT: {event_type}]"

            # Generate speech and audio file in background (voice_loop handles printing & audio)
            res = query_myra_llm(prompt=prompt, initial_emote=initial_emote, output_file="interruption.wav")
            spoken_reply, audio_file = res
            raw_reply = getattr(res, "raw_reply", spoken_reply)

            # Queue proactive event for voice_loop terminal to print and play audio
            proactive_events_queue.put({
                "event_type": event_type,
                "text": spoken_reply,
                "raw_text": raw_reply,
                "audio_file": audio_file,
                "timestamp": time.time()
            })

        except Exception as e:
            pass
        finally:
            is_proactive_speaking = False
            system_status = "IDLE"

    threading.Thread(target=_worker, daemon=True).start()

# -------------------- ON-DEMAND ROOM INSPECTION (VLM) --------------------

def inspect_room_vlm(frame=None, custom_prompt=None):
    """Captures a frame and queries Ollama VLM (moondream) for a concise room/object summary."""
    global current_frame
    target_frame = frame if frame is not None else current_frame
    if target_frame is None:
        return "Camera frame is currently unavailable."

    try:
        h, w = target_frame.shape[:2]
        if w > 800:
            scale = 800 / float(w)
            target_frame = cv2.resize(target_frame, (800, int(h * scale)))

        _, buffer = cv2.imencode('.jpg', target_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        vlm_prompt = custom_prompt or "Describe what is visible in this room, focusing on objects, people, and what the person in front of the camera is holding or doing. Keep it concise (2 sentences max)."

        payload = {
            "model": "moondream",
            "prompt": vlm_prompt,
            "images": [img_b64],
            "stream": False
        }

        res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=45)
        if res.status_code == 200:
            summary = res.json().get("response", "").strip()
            print(f"--- [VLM MOONDREAM SUMMARY]: {summary} ---")
            return summary
        else:
            return "Unable to inspect the room clearly at the moment."
    except Exception as e:
        print(f"--- [VLM ERROR]: {e} ---")
        return "Visual inspection timed out or encountered an error."

# -------------------- ASYNCHRONOUS HIGH-PERFORMANCE VISION ENGINE --------------------

class ThreadedCamera:
    """
    Dedicated background frame capture thread.
    Absorbs DirectShow hardware driver latency, eliminates internal frame buffering lag,
    and provides instant non-blocking frame retrieval for silky-smooth 30-60 FPS UI rendering.
    """
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap = cv2.VideoCapture(src)

        # Configure high-speed MJPG stream and low buffer to eliminate lag
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.thread.start()

    def _capture_worker(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame.copy()

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.cap.release()

class AsyncVisionEngine:
    """
    Decouples real-time 30+ FPS camera display from background AI inference.
    Prevents any camera stuttering or frame drops.
    """
    def __init__(self):
        self.lock = threading.Lock()
        self.running = True
        self.latest_raw_frame = None
        self.new_frame_event = threading.Event()

        # Thread-safe cached detection states for UI display
        self.cached_faces = []
        self.cached_objects = []
        self.cached_person_count = 0
        self.cached_target_name = "Mallu"
        self.cached_objects_str = ""

    def ai_worker_loop(self, yolo, face_app, presence_tracker):
        """Continuously processes latest frame in background without blocking display loop."""
        global current_visual_state, current_detected_objects, primary_target_name

        try:
            torch.set_num_threads(4)
        except Exception:
            pass

        while self.running:
            if not self.new_frame_event.wait(timeout=0.1):
                continue
            self.new_frame_event.clear()

            with self.lock:
                if self.latest_raw_frame is None:
                    continue
                frame = self.latest_raw_frame.copy()

            orig_h, orig_w = frame.shape[:2]

            # Downscale frame to 320w for ultra-fast YOLO & InsightFace inference (3x speedup on CPU)
            target_w = 320
            scale = target_w / float(orig_w)
            target_h = int(orig_h * scale)
            small_frame = cv2.resize(frame, (target_w, target_h))

            scale_x = orig_w / float(target_w)
            scale_y = orig_h / float(target_h)

            # 1. Real-time YOLO11s Object & Person Tracking
            detected_items_names = []
            detected_objects_boxes = []
            person_count = 0

            try:
                yolo_results = yolo.predict(small_frame, imgsz=320, device="cpu", verbose=False)[0]
                for box in yolo_results.boxes:
                    class_id = int(box.cls[0])
                    item_name = yolo.names[class_id]
                    s_xyxy = box.xyxy[0].cpu().numpy().astype(float)

                    # Scale coordinates back to original frame
                    x1 = int(s_xyxy[0] * scale_x)
                    y1 = int(s_xyxy[1] * scale_y)
                    x2 = int(s_xyxy[2] * scale_x)
                    y2 = int(s_xyxy[3] * scale_y)

                    if item_name == "person":
                        person_count += 1
                    elif item_name in ["cell phone", "keyboard", "cup", "bottle", "book", "backpack", "laptop", "mouse", "glasses"]:
                        clean_name = "phone" if item_name == "cell phone" else item_name
                        detected_items_names.append(clean_name)
                        detected_objects_boxes.append({
                            "name": clean_name,
                            "bbox": [x1, y1, x2, y2]
                        })
            except Exception:
                pass

            detected_items_names = list(set(detected_items_names))
            objects_str = ", ".join(detected_items_names) if detected_items_names else "nothing in hands"

            # 2. InsightFace Face Extraction & Cosine Similarity Matching
            detected_faces = []
            target_name = "Mallu" if person_count > 0 else "None"

            if face_app is not None:
                try:
                    faces = face_app.get(small_frame)
                    for f in faces:
                        emb = f.embedding
                        matched_name, sim = match_face(emb, threshold=FACE_MATCH_THRESHOLD)
                        s_bbox = f.bbox.astype(float)

                        # Scale coordinates back to original frame
                        fx1 = int(s_bbox[0] * scale_x)
                        fy1 = int(s_bbox[1] * scale_y)
                        fx2 = int(s_bbox[2] * scale_x)
                        fy2 = int(s_bbox[3] * scale_y)

                        detected_faces.append({
                            'name': matched_name,
                            'similarity': sim,
                            'embedding': emb,
                            'bbox': [fx1, fy1, fx2, fy2],
                            'is_known': (matched_name != "Unknown")
                        })

                    if detected_faces:
                        known_matches = [df['name'] for df in detected_faces if df['is_known']]
                        target_name = known_matches[0] if known_matches else detected_faces[0]['name']
                except Exception:
                    pass

            # 3. PresenceTracker Updates
            events = presence_tracker.update(detected_faces, person_count)
            for ev_type, ev_data in events:
                trigger_proactive_speech(ev_type, ev_data)

            # 4. Atomically update cached states
            with self.lock:
                self.cached_faces = detected_faces
                self.cached_objects = detected_objects_boxes
                self.cached_person_count = person_count
                self.cached_target_name = target_name
                self.cached_objects_str = objects_str

            current_detected_objects = detected_items_names
            primary_target_name = target_name
            current_visual_state = f"Target in view: {target_name}. People in room: {person_count}. Holding/Nearby Objects: {objects_str}."

            # Subtle throttle to keep CPU balanced
            time.sleep(0.03)

vision_engine = AsyncVisionEngine()
ENABLE_BACKGROUND_CAMERA = False  # Set to True only if background webcam tracking is explicitly desired
active_camera_instance = None
camera_thread = None

def start_camera_background():
    """Starts the camera capture loop in a background thread on demand."""
    global camera_thread, active_camera_instance
    if camera_thread is not None and camera_thread.is_alive():
        return False
    vision_engine.running = True
    camera_thread = threading.Thread(target=vision_monitor_loop, daemon=True)
    camera_thread.start()
    system_monitor.log("EYES", "INFO", "Webcam hardware started by user.")
    return True

def stop_camera_background():
    """Immediately stops the camera loop and releases the webcam hardware."""
    global camera_thread, active_camera_instance, current_frame
    vision_engine.running = False
    if active_camera_instance is not None:
        try:
            active_camera_instance.release()
        except Exception:
            pass
        active_camera_instance = None
    system_monitor.update_frame(None)
    current_frame = None
    cv2.destroyAllWindows()
    system_monitor.log("EYES", "INFO", "Webcam hardware stopped and released.")
    return True

def vision_monitor_loop():
    """Main camera capture & render loop running at silky-smooth 30+ FPS."""
    global current_frame, system_status, face_app_instance, active_camera_instance

    print("--- [EYES] Initializing High-Performance Vision Engine (YOLO11s + InsightFace)... ---")
    yolo = YOLO("yolo11s.pt")
    
    face_app = None
    if HAS_INSIGHTFACE:
        try:
            # Optimize: only load detection and recognition modules
            face_app = FaceAnalysis(name='buffalo_s', allowed_modules=['detection', 'recognition'], providers=['CPUExecutionProvider'])
            face_app.prepare(ctx_id=-1, det_size=(320, 320))
            face_app_instance = face_app
            print("--- [EYES] InsightFace ArcFace + SCRFD Ready! ---")
            auto_migrate_existing_face_images(face_app)
        except Exception as e:
            print(f"--- [EYES] InsightFace init notice: {e} ---")
            face_app = None

    load_known_faces()
    presence_tracker = PresenceTracker()

    # Start background AI worker thread
    ai_thread = threading.Thread(
        target=vision_engine.ai_worker_loop,
        args=(yolo, face_app, presence_tracker),
        daemon=True
    )
    ai_thread.start()

    cam = ThreadedCamera(0)
    active_camera_instance = cam
    if not cam.isOpened():
        print("--- [EYES] Camera not found or busy. ---")
        active_camera_instance = None
        return

    try:
        cv2.namedWindow("Myra's Eyes", cv2.WINDOW_NORMAL)

        while vision_engine.running:
            ret, frame = cam.read()
            if not ret or frame is None:
                time.sleep(0.01)
                continue

            current_frame = frame
            display_frame = frame.copy()
            h, w = frame.shape[:2]

            # Feed latest frame to AI worker
            with vision_engine.lock:
                vision_engine.latest_raw_frame = frame
            vision_engine.new_frame_event.set()

            # Grab cached detection results atomically
            with vision_engine.lock:
                faces = list(vision_engine.cached_faces)
                objects = list(vision_engine.cached_objects)
                person_count = vision_engine.cached_person_count
                target_name = vision_engine.cached_target_name
                objects_str = vision_engine.cached_objects_str

            # 1. Render Object Bounding Boxes (Subtle Cyan/Gray, 1px thickness)
            for obj in objects:
                clean_name = obj["name"]
                x1, y1, x2, y2 = obj["bbox"]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (220, 180, 50), 1)
                pill_w = len(clean_name) * 8 + 8
                cv2.rectangle(display_frame, (x1, max(0, y1 - 16)), (x1 + pill_w, max(16, y1)), (220, 180, 50), -1)
                cv2.putText(display_frame, clean_name, (x1 + 4, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            # 2. Render Face Bounding Boxes (Green for Known, Yellow/Amber for Unknown, 1px thickness)
            for f in faces:
                matched_name = f["name"]
                sim = f["similarity"]
                is_known = f["is_known"]
                fx1, fy1, fx2, fy2 = f["bbox"]

                box_color = (0, 255, 0) if is_known else (0, 200, 255)
                cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), box_color, 1)

                label_text = f"{matched_name} | {sim:.2f}" if is_known else f"Unknown | {sim:.2f}"
                pill_w = len(label_text) * 8 + 10
                cv2.rectangle(display_frame, (fx1, max(0, fy1 - 18)), (fx1 + pill_w, max(18, fy1)), box_color, -1)
                cv2.putText(display_frame, label_text, (fx1 + 4, max(13, fy1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1)

            # 3. Render Clean Top HUD Banner Overlay (Ultra-fast slice alpha blend, 100x faster, zero full-frame copies)
            banner = display_frame[0:32, 0:w]
            dark_bar = np.full_like(banner, (20, 20, 20))
            cv2.addWeighted(dark_bar, 0.70, banner, 0.30, 0, banner)
            display_frame[0:32, 0:w] = banner

            # Header Text
            if presentation_manager.is_presenting:
                ppt_stat = presentation_manager.get_status()
                hud_text = f"STATUS: PRESENTING (Slide {ppt_stat['current_slide']}/{ppt_stat['total_slides']}) | TARGET: {target_name} | PEOPLE: {person_count} | OBJS: {objects_str[:25]}"
            else:
                hud_text = f"STATUS: {system_status} | TARGET: {target_name} | PEOPLE: {person_count} | OBJS: {objects_str[:30]}"
            cv2.putText(display_frame, hud_text, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

            # Update monitor stream frame
            system_monitor.update_frame(display_frame)

            cv2.imshow("Myra's Eyes", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        vision_engine.running = False
        cam.release()
        active_camera_instance = None
        cv2.destroyAllWindows()
        print("--- [EYES] Camera hardware released successfully. ---")

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

def warmup_ollama():
    """Warms up Ollama in background so first user request doesn't time out."""
    ensure_ollama_running()
    try:
        print("--- [BRAIN] Warming up Ollama models in VRAM... ---")
        requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "myra", "messages": [{"role": "user", "content": "hi"}], "stream": False},
            timeout=40
        )
        print("--- [BRAIN] Ollama is warm and ready! ---")
    except Exception as e:
        print(f"--- [BRAIN] Ollama warmup note: {e} ---")

@asynccontextmanager
async def lifespan(app: FastAPI):
    system_monitor.log("SYSTEM", "SUCCESS", "FastAPI core online on port 8000. Camera hardware in standby mode.")
    if ENABLE_BACKGROUND_CAMERA:
        start_camera_background()
    threading.Thread(target=warmup_ollama, daemon=True).start()
    yield
    system_monitor.log("SYSTEM", "WARN", "Shutting down backend services.")
    stop_camera_background()

app = FastAPI(title="Myra AI Assistant Backend", version="2.0", lifespan=lifespan)

# Mount Frontend Static Assets
if os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

class ChatRequest(BaseModel):
    message: str

class LearnPersonRequest(BaseModel):
    name: str
    relationship: str = "Friend"

class RegisterFaceRequest(BaseModel):
    name: str

class PresentationRequest(BaseModel):
    action: str = "start"
    file_path: str = None
    slide_num: int = None

@app.get("/status")
def get_status():
    """Lightweight status endpoint for frontend/voice loop coordination."""
    ppt_stat = presentation_manager.get_status()

    # Check for queued proactive speech/stranger alerts
    proactive_ev = None
    if not proactive_events_queue.empty():
        try:
            proactive_ev = proactive_events_queue.get_nowait()
        except queue.Empty:
            proactive_ev = None

    return {
        "status": system_status,
        "awaiting_name_confirmation": awaiting_name_confirmation,
        "is_proactive_speaking": is_proactive_speaking,
        "target": primary_target_name,
        "objects": current_detected_objects,
        "is_presenting": ppt_stat["is_presenting"],
        "current_slide": ppt_stat["current_slide"],
        "total_slides": ppt_stat["total_slides"],
        "slide_title": ppt_stat["slide_title"],
        "proactive_event": proactive_ev
    }

@app.post("/present_ppt")
def present_ppt_endpoint(req: PresentationRequest):
    action = req.action.lower()
    if action == "start":
        speech_reply, audio_file = presentation_manager.start_presentation(filepath=req.file_path, query_llm_fn=query_myra_llm)
    elif action == "next":
        speech_reply, audio_file = presentation_manager.next_slide(query_llm_fn=query_myra_llm)
    elif action == "prev":
        speech_reply, audio_file = presentation_manager.prev_slide(query_llm_fn=query_myra_llm)
    elif action == "goto":
        slide_n = req.slide_num if req.slide_num is not None else 1
        speech_reply, audio_file = presentation_manager.goto_slide(slide_n, query_llm_fn=query_myra_llm)
    elif action in ["explain", "read"]:
        speech_reply, audio_file = presentation_manager.present_current_slide(query_llm_fn=query_myra_llm)
    elif action == "stop":
        speech_reply, audio_file = presentation_manager.stop_presentation(query_llm_fn=query_myra_llm)
    elif action == "status":
        return {"status": presentation_manager.get_status()}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown presentation action: {action}")

    return {
        "response": speech_reply,
        "audio_file": audio_file,
        "status": presentation_manager.get_status()
    }

@app.post("/chat")
def chat(request: ChatRequest):
    global current_visual_state, last_interaction_time, system_status
    global awaiting_name_confirmation, pending_stranger_embedding

    last_interaction_time = time.time()
    user_message = request.message.strip()
    user_lower = user_message.lower()
    print(f"\n--- [USER] {user_message} ---")
    system_monitor.log("BRAIN", "INFO", f"User input: '{user_message[:70]}'")

    # A. Check for Autonomous Presentation Voice Commands
    # 1. Stop presentation
    if presentation_manager.is_presenting and any(k in user_lower for k in ["stop presentation", "end presentation", "close presentation", "close slides", "stop presenting"]):
        spoken_reply, audio_file = presentation_manager.stop_presentation(query_llm_fn=query_myra_llm)
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("user", user_message))
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("assistant", spoken_reply))
                conn.commit()
        except Exception:
            pass
        return {"response": spoken_reply, "raw_response": spoken_reply, "audio_file": audio_file, "is_presenting": False}

    # 2. Next slide / Continue presentation
    is_next_cmd = any(k in user_lower for k in ["next slide", "next page", "continue presentation", "agla slide"]) or (
        presentation_manager.is_presenting and any(user_lower.strip() == w for w in ["next", "continue", "move on", "proceed", "go ahead", "next one"])
    )
    if is_next_cmd and presentation_manager.is_presenting:
        spoken_reply, audio_file = presentation_manager.next_slide(query_llm_fn=query_myra_llm)
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("user", user_message))
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("assistant", spoken_reply))
                conn.commit()
        except Exception:
            pass
        return {"response": spoken_reply, "raw_response": spoken_reply, "audio_file": audio_file, "is_presenting": presentation_manager.is_presenting}

    # 3. Previous slide
    if presentation_manager.is_presenting and any(k in user_lower for k in ["previous slide", "prev slide", "go back", "last slide", "pichla slide"]):
        spoken_reply, audio_file = presentation_manager.prev_slide(query_llm_fn=query_myra_llm)
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("user", user_message))
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("assistant", spoken_reply))
                conn.commit()
        except Exception:
            pass
        return {"response": spoken_reply, "raw_response": spoken_reply, "audio_file": audio_file, "is_presenting": True}

    # 4. Start presentation (Flexible matching for PPT, PPD, PBT, PDF, Slides, Project)
    is_start_ppt = any(k in user_lower for k in [
        "start presentation", "present the ppt", "present the ppd", "present the pbt", "present our project",
        "present the slides", "start presenting", "present ppt", "present ppd", "present pbt", "present pdf",
        "present the pdf", "present project", "presentation start", "shuru karo presentation"
    ]) or (
        any(v in user_lower for v in ["present", "start", "show", "open", "run", "explain"]) and
        any(n in user_lower for n in ["ppt", "ppd", "pbt", "pdf", "presentation", "slides", "deck", "project"])
    )
    if is_start_ppt:
        spoken_reply, audio_file = presentation_manager.start_presentation(query_llm_fn=query_myra_llm)
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("user", user_message))
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("assistant", spoken_reply))
                conn.commit()
        except Exception:
            pass
        return {"response": spoken_reply, "raw_response": spoken_reply, "audio_file": audio_file, "is_presenting": True}

    # B. Classroom Mass Introduction Trigger
    if any(k in user_lower for k in ["meet my class", "introduce you to my class", "meet the class", "say hi to my class", "introduce yourself to everyone", "introduce yourself to my class", "meet everyone in class", "meet all of them"]):
        headcount = max(vision_engine.cached_person_count, 1)
        trigger_vmc_expression("Surprised", 1.0, duration=2.0)
        def _delay_smug():
            time.sleep(2.0)
            trigger_vmc_expression("Fun", 0.9, duration=3.0)
        threading.Thread(target=_delay_smug, daemon=True).start()

        prompt = (
            f"[EVENT: Mallu is introducing you to his entire college class on a big screen. "
            f"There are {headcount} students staring at you. "
            f"In your smug, teasing, playful waifu personality, react to the huge crowd and ask Mallu if he brought an entire fan club.]"
        )
        spoken_reply, audio_file = query_myra_llm(prompt=prompt, initial_emote="smug")
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("user", user_message))
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("assistant", spoken_reply))
                conn.commit()
        except Exception:
            pass
        return {"response": spoken_reply, "raw_response": spoken_reply, "audio_file": audio_file}

    # C. Faculty Politeness & Title Detection
    faculty_match = re.search(r"\b(professor|prof\.?|doctor|dr\.?|dean|sir|ma'am|madam)\b(?:\s+([A-Za-z]+))?", user_message, re.IGNORECASE)
    is_faculty_intro = faculty_match and any(k in user_lower for k in ["meet", "this is", "introduce", "say hi to", "greeting", "hello", "namaste"])
    if is_faculty_intro:
        raw_title = faculty_match.group(1).capitalize()
        if raw_title.lower().startswith("prof"):
            raw_title = "Professor"
        elif raw_title.lower().startswith("dr"):
            raw_title = "Doctor"
        elif raw_title.lower() in ["ma'am", "madam"]:
            raw_title = "Ma'am"
        elif raw_title.lower() == "sir":
            raw_title = "Sir"

        name_part = faculty_match.group(2).capitalize() if faculty_match.group(2) else ""
        faculty_full_title = f"{raw_title} {name_part}".strip()

        # Register faculty face if available
        if pending_stranger_embedding is not None:
            register_new_face(faculty_full_title, pending_stranger_embedding)
            awaiting_name_confirmation = False
            pending_stranger_embedding = None
        else:
            with vision_engine.lock:
                unknown_faces = [f for f in vision_engine.cached_faces if not f['is_known']]
                if unknown_faces:
                    register_new_face(faculty_full_title, unknown_faces[0]['embedding'])

        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO known_people (name, relationship, notes)
                    VALUES (?, ?, ?)
                    ON CONFLICT(name) DO UPDATE SET relationship=excluded.relationship
                """, (faculty_full_title, f"College {raw_title}", f"Registered during college presentation on {datetime.now().strftime('%Y-%m-%d')}"))
                conn.commit()
        except Exception:
            pass

        trigger_vmc_expression("Joy", 0.8, duration=3.0)
        prompt = (
            f"[EVENT: Mallu is introducing you to his college faculty/professor, {faculty_full_title}. "
            f"You MUST be extremely polite, respectful, and charming to make a great impression on the teacher, "
            f"while still having a sweet, cute smile. Greet {faculty_full_title} warmly.]"
        )
        spoken_reply, audio_file = query_myra_llm(prompt=prompt, initial_emote="smile")
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("user", user_message))
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("assistant", spoken_reply))
                conn.commit()
        except Exception:
            pass
        return {"response": spoken_reply, "raw_response": spoken_reply, "audio_file": audio_file}

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
    if any(k in user_message.lower() for k in ["presentation", "ppt", "slide", "deck"]):
        target_ppt = None
        if os.path.exists("myra.pptx"):
            target_ppt = "myra.pptx"
        else:
            for f in os.listdir("."):
                if f.endswith(".pptx") and not f.startswith("~$"):
                    target_ppt = f
                    break
        if target_ppt:
            ppt_text = parse_presentation(target_ppt)
            if ppt_text:
                ppt_context = f"\n[ATTACHED PRESENTATION FILE: {target_ppt}]\n{ppt_text}\n"

    # 2.5 Check for Direct Mallu Face Identification / Self-Enrollment
    is_mallu_self_id = any(phrase in user_lower for phrase in [
        "i am mallu", "it's me mallu", "its me mallu", "it is me mallu",
        "remember my face", "learn my face", "this is mallu", "recognize me",
        "mai hu mallu", "main mallu hu", "who am i", "pehechana", "pechana"
    ])
    if is_mallu_self_id:
        target_emb = None
        if pending_stranger_embedding is not None:
            target_emb = pending_stranger_embedding
            awaiting_name_confirmation = False
            pending_stranger_embedding = None
        elif current_frame is not None and face_app_instance is not None:
            faces = face_app_instance.get(current_frame)
            if faces:
                target_emb = faces[0].embedding
        elif vision_engine.cached_faces:
            target_emb = vision_engine.cached_faces[0]['embedding']

        if target_emb is not None:
            register_new_face("Mallu", target_emb)
            primary_target_name = "Mallu"

        trigger_vmc_expression("Joy", 1.0, duration=3.0)
        prompt = "[EVENT: Mallu just told you that it's him in front of the camera. In your slightly flustered, sweet, affectionate girlfriend persona, blush, apologize playfully for not recognizing him under the current lighting, and tell him you've locked his handsome face into memory now.]"
        spoken_reply, audio_file = query_myra_llm(prompt=prompt, initial_emote="blush")
        try:
            with get_db() as conn:
                cursor = conn.cursor()
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("user", user_message))
                cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("assistant", spoken_reply))
                conn.commit()
        except Exception:
            pass
        return {"response": spoken_reply, "raw_response": spoken_reply, "audio_file": audio_file}

    # 3. Check for Conversational Name Confirmation for Stranger
    name_intro_context = ""
    if awaiting_name_confirmation and pending_stranger_embedding is not None:
        # First check: Is this actually Mallu?
        matched_who, sim_who = match_face(pending_stranger_embedding, threshold=0.40)
        if matched_who == "Mallu":
            register_new_face("Mallu", pending_stranger_embedding)
            primary_target_name = "Mallu"
            awaiting_name_confirmation = False
            pending_stranger_embedding = None
            name_intro_context = "\n[SYSTEM NOTIFICATION: You looked closer and realized this is Mallu himself! Tease him about how the lighting confused you for a second.]\n"
        else:
            name_match = re.search(r"(?:this is|meet(?:\s+my friend)?|his name is|her name is|he is|she is|name is|friend)\s+([A-Za-z]+)", user_message, re.IGNORECASE)
            if not name_match:
                # Direct answer to "who is this": e.g. "Rahul", "It's Rahul", "He is Rahul"
                direct_match = re.match(r"^(?:he's|she's|it's|this is|his name is)?\s*([A-Za-z]+)$", user_message.strip(), re.IGNORECASE)
                if direct_match and direct_match.group(1).lower() not in ["yes", "no", "what", "who", "why", "hey", "hello", "hi", "ok", "okay", "sure", "wait", "hmm", "neither", "leave", "stop", "nothing", "nobody"]:
                    name_match = direct_match

            if name_match:
                new_name = name_match.group(1).capitalize()
                if new_name.lower() in ["mallu", "me", "myself"]:
                    register_new_face("Mallu", pending_stranger_embedding)
                    primary_target_name = "Mallu"
                    name_intro_context = "\n[SYSTEM NOTIFICATION: It's Mallu! You recognized him and saved his face.]\n"
                elif new_name.lower() not in ["no", "nobody", "nothing", "none", "someone", "ignore", "yes", "what", "who", "why", "hey", "hello", "hi", "ok", "okay", "sure", "wait", "hmm"]:
                    registered = register_new_face(new_name, pending_stranger_embedding)
                    if registered:
                        name_intro_context = f"\n[SYSTEM NOTIFICATION: Mallu just introduced the stranger as '{new_name}'. You have successfully saved and remembered {new_name}'s face! Greet {new_name} warmly and playfully in your cute, slightly teasing style.]\n"
                awaiting_name_confirmation = False
                pending_stranger_embedding = None
            elif any(k in user_message.lower() for k in ["nobody", "no one", "ignore", "leave it", "nobody special"]):
                awaiting_name_confirmation = False
                pending_stranger_embedding = None

    # 4. Check for On-Demand VLM Room Inspection
    vlm_context = ""
    is_vlm_query = any(k in user_message.lower() for k in [
        "look around", "what is in my room", "what's in my room", "what do you see",
        "what am i holding", "what's in my hand", "describe the room", "inspect room",
        "what do you notice", "see around", "take a look", "kya dikh raha hai", "room me kya hai"
    ])
    if is_vlm_query:
        print("--- [BRAIN] User requested visual room inspection, querying Moondream VLM... ---")
        vlm_desc = inspect_room_vlm()
        if vlm_desc:
            vlm_context = f"\n[VLM DETAILED ROOM OBSERVATION: {vlm_desc}]\n"

    # 5. Build messages array for Ollama
    system_block = {"role": "system", "content": MYRA_SYSTEM_PROMPT}
    current_prompt = f"[VISION: {current_visual_state}]{ppt_context}{vlm_context}{name_intro_context}\n{user_message}"
    
    messages = [system_block]
    messages.extend(history)
    messages.append({"role": "user", "content": current_prompt})

    # 6. Query Ollama LLaMA 3
    raw_reply = None
    for attempt in range(2):
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
            if ollama_res.status_code == 200:
                raw_reply = ollama_res.json()["message"]["content"]
                break
        except Exception as e:
            print(f"--- [BRAIN] Ollama Query Attempt {attempt+1} Notice: {e} ---")
            if attempt == 0 and ensure_ollama_running():
                continue

    if not raw_reply:
        raw_reply = "[EMOTE: pout] Ugh, my brain lagged for a second... Ask me again, Mallu!"

    # 7. Apply VMC Emotion and generate Sweet Voice
    spoken_reply = apply_emotion_tag(raw_reply)
    print(f"--- [MYRA] {raw_reply} ---")
    system_monitor.log("BRAIN", "SUCCESS", f"Myra reply generated: '{spoken_reply[:70]}'")

    # 8. Save to SQLite Memory
    try:
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("user", user_message))
            cursor.execute("INSERT INTO conversations (role, content) VALUES (?, ?)", ("assistant", raw_reply))
            conn.commit()
    except Exception as e:
        print(f"Memory Save Error: {e}")

    # 9. Generate Audio (Original Silero)
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

@app.post("/inspect_room")
def inspect_room_endpoint():
    """Manual on-demand VLM room inspection endpoint."""
    desc = inspect_room_vlm()
    return {"status": "success", "description": desc}

@app.post("/register_face")
def register_face_endpoint(req: RegisterFaceRequest):
    """Registers the currently visible face in the webcam to SQLite known_faces."""
    global current_frame, face_app_instance
    if current_frame is None:
        raise HTTPException(status_code=400, detail="No camera frame available.")
    if face_app_instance is None:
        raise HTTPException(status_code=500, detail="InsightFace not initialized.")
    faces = face_app_instance.get(current_frame)
    if not faces:
        raise HTTPException(status_code=400, detail="No face detected in current frame.")
    success = register_new_face(req.name, faces[0].embedding)
    return {"status": "success" if success else "failed", "name": req.name}

@app.post("/learn_face")
def learn_face(req: LearnPersonRequest):
    """Saves current webcam snapshot and registers face embedding."""
    global current_frame, face_app_instance
    if current_frame is None:
        raise HTTPException(status_code=400, detail="No webcam frame available.")
    
    clean_name = req.name.strip().lower().replace(" ", "_")
    target_path = os.path.join(DB_PATH, f"{clean_name}.jpg")
    cv2.imwrite(target_path, current_frame)

    if face_app_instance is not None:
        faces = face_app_instance.get(current_frame)
        if faces:
            register_new_face(req.name, faces[0].embedding)

    with get_db() as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO known_people (name, relationship, notes)
            VALUES (?, ?, ?)
            ON CONFLICT(name) DO UPDATE SET relationship=excluded.relationship
        """, (req.name.strip(), req.relationship, f"Learned on {datetime.now().strftime('%Y-%m-%d')}"))
        conn.commit()

    return {"status": "success", "message": f"Learned {req.name}'s face successfully!"}

# -------------------- MONITOR & DIAGNOSTIC SYSTEM ROUTES --------------------

class SkillTestRequest(BaseModel):
    skill: str

@app.get("/", response_class=FileResponse)
@app.get("/monitor", response_class=FileResponse)
def serve_monitor():
    """Serves Myra's Neural Subsystem & Diagnostic Monitoring Frontend."""
    index_file = os.path.join("frontend", "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return HTMLResponse("<h2>Myra Monitor index.html not found. Ensure frontend/ is present.</h2>")

@app.get("/api/monitor/skills")
def monitor_skills_endpoint():
    """Returns comprehensive real-time telemetry across all 8 subsystems."""
    silero_loaded = silero_model is not None
    person_cnt = vision_engine.cached_person_count if hasattr(vision_engine, "cached_person_count") else 0
    return system_monitor.get_full_status(
        vision_engine=vision_engine,
        primary_target_name=primary_target_name,
        person_count=person_cnt,
        detected_objects=current_detected_objects,
        presentation_manager=presentation_manager,
        silero_model_loaded=silero_loaded
    )

@app.post("/api/monitor/test_skill")
def test_skill_endpoint(req: SkillTestRequest):
    """Executes live diagnostic test on the requested subsystem."""
    return system_monitor.test_skill(req.skill)

@app.get("/api/monitor/logs")
def monitor_logs_endpoint(limit: int = 100, level: str = None, subsystem: str = None):
    """Fetches recent subsystem logs with optional filtering."""
    return system_monitor.get_logs(limit=limit, level=level, subsystem=subsystem)

def generate_mjpeg_stream():
    """Yields live MJPEG video stream of Myra's Optical Vision Feed."""
    while True:
        frame = system_monitor.latest_display_frame
        if frame is None:
            # Standby animation canvas
            blank = np.zeros((320, 520, 3), dtype=np.uint8)
            cv2.putText(blank, "MYRA OPTICAL FEED - STANDBY", (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 242, 254), 1)
            cv2.putText(blank, "Vision engine is starting or camera 0 is idle...", (70, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (120, 140, 160), 1)
            ret, buffer = cv2.imencode('.jpg', blank)
            time.sleep(0.1)
        else:
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            time.sleep(0.033)

        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.get("/api/monitor/video_feed")
def video_feed_endpoint():
    """Streams live optical perception view with bounding boxes as MJPEG."""
    return StreamingResponse(generate_mjpeg_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

# Camera on-demand lifecycle controls
@app.post("/api/camera/start")
def start_camera_endpoint():
    """Starts the optical camera hardware stream on demand."""
    started = start_camera_background()
    return {"status": "started" if started else "already_running"}

@app.post("/api/camera/stop")
def stop_camera_endpoint():
    """Stops the optical camera hardware stream and completely releases webcam hardware."""
    stop_camera_background()
    return {"status": "stopped", "message": "Camera hardware released."}

@app.get("/api/camera/status")
def camera_status_endpoint():
    """Checks whether the camera hardware is currently active."""
    active = vision_engine.running and (active_camera_instance is not None)
    return {"active": active}

# -------------------- ENTRY POINT --------------------

if __name__ == "__main__":
    import uvicorn
    print("--- [SYSTEM] Starting Myra Backend on Port 8000... ---")
    uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False)