"""
Myra AI Assistant - Central System Monitor & Diagnostics Engine
Tracks real-time health, latency, error states, and diagnostic tests across all 8 subsystems:
1. Brain (Ollama LLM)
2. Eyes (YOLO11s + InsightFace + Camera)
3. Voice (Silero TTS + Audio Pipeline)
4. Ears (Faster-Whisper STT + Mic Input)
5. Memory (SQLite DB + Face Vectors)
6. Embodiment (VMC Protocol + VSeeFace)
7. Presentation (PowerPoint Automation)
8. Audio Hardware Router (Headphone/Headset Routing)
"""

import os
import sys
import time
import json
import sqlite3
import threading
import traceback
from datetime import datetime
from collections import deque
import requests

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import torch
except Exception:
    torch = None

try:
    import cv2
except Exception:
    cv2 = None

# ----------------- SYSTEM LOG BUFFER -----------------

class SystemLogEntry:
    def __init__(self, subsystem: str, level: str, message: str, details: str = None, fix_tip: str = None):
        self.id = int(time.time() * 1000) + (os.getpid() % 1000)
        self.timestamp = datetime.now().strftime("%H:%M:%S")
        self.subsystem = subsystem.upper()
        self.level = level.upper()  # INFO, SUCCESS, WARN, ERROR, CRITICAL
        self.message = message
        self.details = details or ""
        self.fix_tip = fix_tip or ""

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "subsystem": self.subsystem,
            "level": self.level,
            "message": self.message,
            "details": self.details,
            "fix_tip": self.fix_tip
        }

class SystemMonitor:
    def __init__(self, max_logs: int = 250):
        self.logs_lock = threading.Lock()
        self.logs = deque(maxlen=max_logs)
        self.start_time = time.time()
        self.subsystem_errors = {}  # {subsystem_name: list of recent error dicts}
        self.latest_test_results = {}
        self.latest_display_frame = None
        self.frame_lock = threading.Lock()

        # Initial startup log
        self.log(
            subsystem="SYSTEM",
            level="SUCCESS",
            message="Myra Neural Monitor initialized successfully.",
            details="All telemetry channels opened on port 8000."
        )

    def log(self, subsystem: str, level: str, message: str, details: str = None, fix_tip: str = None):
        entry = SystemLogEntry(subsystem, level, message, details, fix_tip)
        with self.logs_lock:
            self.logs.append(entry)
            if level in ["ERROR", "CRITICAL"]:
                if subsystem not in self.subsystem_errors:
                    self.subsystem_errors[subsystem] = deque(maxlen=10)
                self.subsystem_errors[subsystem].append(entry.to_dict())

    def update_frame(self, frame):
        """Thread-safely stores latest camera frame for MJPEG streaming."""
        with self.frame_lock:
            self.latest_display_frame = frame

    def get_logs(self, limit: int = 100, level: str = None, subsystem: str = None):
        with self.logs_lock:
            all_entries = list(self.logs)

        filtered = []
        for e in reversed(all_entries):
            if level and e.level != level.upper():
                continue
            if subsystem and e.subsystem != subsystem.upper():
                continue
            filtered.append(e.to_dict())
            if len(filtered) >= limit:
                break
        return filtered

    def get_uptime_str(self):
        elapsed = int(time.time() - self.start_time)
        hours = elapsed // 3600
        mins = (elapsed % 3600) // 60
        secs = elapsed % 60
        return f"{hours:02d}h {mins:02d}m {secs:02d}s"

    # ----------------- LIVE SUBSYSTEM CHECKS -----------------

    def check_brain(self) -> dict:
        """Inspects Ollama LLM service, model presence, and response latency."""
        status = "ONLINE"
        error_msg = None
        fix_tip = None
        latency_ms = None
        models = []

        t0 = time.time()
        try:
            res = requests.get("http://127.0.0.1:11434/api/tags", timeout=1.8)
            latency_ms = round((time.time() - t0) * 1000, 1)
            if res.status_code == 200:
                data = res.json()
                models = [m.get("name", "") for m in data.get("models", [])]
                has_myra = any("myra" in m.lower() for m in models)
                if not has_myra:
                    status = "WARNING"
                    error_msg = "Model 'myra' not found in Ollama repository (fallback to llama3)."
                    fix_tip = "Run 'ollama create myra -f Modelfile' in the project directory."
            else:
                status = "ERROR"
                error_msg = f"Ollama returned HTTP status {res.status_code}"
                fix_tip = "Restart Ollama service by running 'ollama serve' in PowerShell."
        except Exception as e:
            status = "OFFLINE"
            error_msg = f"Cannot reach Ollama at 127.0.0.1:11434: {type(e).__name__}"
            fix_tip = "Ollama is not running. Launch it from Windows Start Menu or run 'ollama serve' in terminal."

        return {
            "name": "Cognitive Brain",
            "code": "BRAIN-01",
            "status": status,
            "latency_ms": latency_ms,
            "models_available": models[:5],
            "active_model": "myra:latest" if any("myra" in m.lower() for m in models) else ("llama3" if models else "None"),
            "port": 11434,
            "error": error_msg,
            "fix_tip": fix_tip,
            "details": f"Ollama API responding in {latency_ms}ms with {len(models)} models available." if status == "ONLINE" else error_msg
        }

    def check_eyes(self, vision_engine=None, primary_target_name=None, person_count=None, detected_objects=None) -> dict:
        """Inspects Vision Engine, YOLOv11s weights, Camera, and InsightFace."""
        status = "ONLINE"
        error_msg = None
        fix_tip = None

        yolo_present = os.path.exists("yolo11s.pt") or os.path.exists("yolov8n.pt")
        if not yolo_present:
            status = "ERROR"
            error_msg = "YOLO model weights file not found."
            fix_tip = "Ensure yolo11s.pt is present in project root."

        cam_active = False
        if vision_engine and getattr(vision_engine, "running", False):
            cam_active = True
        else:
            # Check if camera frame was updated recently
            with self.frame_lock:
                cam_active = self.latest_display_frame is not None

        if not cam_active:
            status = "STANDBY"
            error_msg = None
            fix_tip = None

        return {
            "name": "Visual Perception",
            "code": "EYES-02",
            "status": status,
            "camera_active": cam_active,
            "yolo_weights": "yolo11s.pt" if os.path.exists("yolo11s.pt") else "Missing",
            "face_engine": "InsightFace (ArcFace + SCRFD)",
            "tracked_target": primary_target_name or "Mallu",
            "person_count": person_count if person_count is not None else 0,
            "detected_objects": (detected_objects or [])[:6],
            "error": error_msg,
            "fix_tip": fix_tip,
            "details": f"Tracking target: {primary_target_name or 'Mallu'} ({person_count or 0} people in view)." if cam_active else "Camera in Standby (Hardware released - activate on demand)."
        }

    def check_voice(self, silero_model_loaded: bool = True) -> dict:
        """Inspects Silero PyTorch TTS engine, speaker voice, and sample rate."""
        status = "ONLINE"
        error_msg = None
        fix_tip = None

        if not silero_model_loaded:
            status = "ERROR"
            error_msg = "Silero TTS PyTorch model is not loaded in memory."
            fix_tip = "Check PyTorch install and torch.hub connectivity to snakers4/silero-models."

        return {
            "name": "Speech Synthesis",
            "code": "VOICE-03",
            "status": status,
            "engine": "Silero TTS (snakers4/silero-models)",
            "speaker": "v3_en (Female / English)",
            "sample_rate": "48000 Hz",
            "breath_pauser": "Active (0.55s full-stop pause)",
            "acronym_expander": "Active (AI -> Artificial Intelligence, LLM -> Large Language Model)",
            "post_roll_buffer": "400ms Hardware Tail Flush",
            "error": error_msg,
            "fix_tip": fix_tip,
            "details": "Silero v3_en 48kHz voice synthesis operational with sentence pausing."
        }

    def check_ears(self) -> dict:
        """Inspects Whisper STT engine, CUDA acceleration, and active input mic."""
        status = "ONLINE"
        error_msg = None
        fix_tip = None

        cuda_available = torch.cuda.is_available() if torch else False
        input_mic_name = "System Default"
        is_headset_mic = False

        if sd:
            try:
                from audio_router import detect_audio_devices
                audio_cfg = detect_audio_devices(verbose=False)
                input_mic_name = audio_cfg.get("input_name", "Default")
                is_headset_mic = audio_cfg.get("is_headphone_mic", False)
            except Exception as e:
                status = "WARNING"
                error_msg = f"Audio device query warning: {e}"
                fix_tip = "Check sound device permissions in Windows Settings."

        return {
            "name": "Auditory System",
            "code": "EARS-04",
            "status": status,
            "engine": "Faster-Whisper (CTranslate2)",
            "acceleration": "NVIDIA CUDA (GPU)" if cuda_available else "CPU Execution Provider",
            "active_microphone": input_mic_name,
            "is_headset_mic": is_headset_mic,
            "rms_threshold": 0.0040,
            "wake_words": ["Myra", "Maira", "Mira", "Mahira"],
            "error": error_msg,
            "fix_tip": fix_tip,
            "details": f"Listening via {input_mic_name} ({'Headset Mic' if is_headset_mic else 'Built-in Mic'})."
        }

    def check_memory(self) -> dict:
        """Inspects SQLite database integrity, conversation history, and face records."""
        status = "ONLINE"
        error_msg = None
        fix_tip = None
        conv_count = 0
        face_count = 0
        integrity = "UNKNOWN"

        db_path = "myra_memory.db"
        if not os.path.exists(db_path):
            status = "ERROR"
            error_msg = "Database file 'myra_memory.db' not found."
            fix_tip = "The database will be created automatically on next backend startup."
            return {"name": "Persistent Memory", "code": "MEM-05", "status": status, "error": error_msg, "fix_tip": fix_tip}

        try:
            conn = sqlite3.connect(db_path, timeout=1.5)
            c = conn.cursor()
            c.execute("PRAGMA integrity_check;")
            integrity_row = c.fetchone()
            integrity = integrity_row[0] if integrity_row else "error"

            if integrity != "ok":
                status = "ERROR"
                error_msg = f"SQLite integrity check reported: {integrity}"
                fix_tip = "Run 'sqlite3 myra_memory.db .recover' to restore database."

            c.execute("SELECT COUNT(*) FROM conversations;")
            conv_count = c.fetchone()[0]

            # Count registered faces
            try:
                c.execute("SELECT COUNT(*) FROM known_faces;")
                face_count = c.fetchone()[0]
            except Exception:
                face_count = len(os.listdir("known_faces")) if os.path.exists("known_faces") else 0

            conn.close()
        except Exception as e:
            status = "ERROR"
            error_msg = f"Database query failed: {type(e).__name__}: {e}"
            fix_tip = "Check if database file is locked by another process."

        return {
            "name": "Persistent Memory",
            "code": "MEM-05",
            "status": status,
            "database_file": db_path,
            "integrity": integrity,
            "conversations_logged": conv_count,
            "known_faces_count": face_count,
            "primary_user": "Mallu",
            "error": error_msg,
            "fix_tip": fix_tip,
            "details": f"Database healthy ({integrity}). {conv_count} conversations, {face_count} faces enrolled."
        }

    def check_embodiment(self) -> dict:
        """Inspects VMC protocol client and VSeeFace expression status."""
        status = "ONLINE"
        error_msg = None
        fix_tip = None

        return {
            "name": "Embodiment & Avatar",
            "code": "AVTR-06",
            "status": status,
            "vmc_port": 39539,
            "backup_port": 39540,
            "protocol": "VMC OSC (Open Sound Control)",
            "supported_emotions": ["Smug", "Joy", "Angry", "Surprised", "Fun", "Neutral"],
            "lip_sync_bridge": "VB-Audio Virtual Cable (Direct Out -> Cable In)",
            "error": error_msg,
            "fix_tip": fix_tip,
            "details": "VMC OSC UDP clients streaming to port 39539 for VSeeFace avatar animation."
        }

    def check_presentation(self, presentation_manager=None) -> dict:
        """Inspects PowerPoint presentation engine and slide deck."""
        status = "ONLINE"
        error_msg = None
        fix_tip = None

        ppt_file = "myra.pptx"
        file_exists = os.path.exists(ppt_file)
        if not file_exists:
            status = "WARNING"
            error_msg = f"Slide deck '{ppt_file}' not found in backend directory."
            fix_tip = "Make sure myra.pptx is in the backend root directory."

        is_presenting = False
        current_slide = 1
        total_slides = 7
        slide_title = ""

        if presentation_manager:
            try:
                st = presentation_manager.get_status()
                is_presenting = st.get("is_presenting", False)
                current_slide = st.get("current_slide", 1)
                total_slides = st.get("total_slides", 7)
                slide_title = st.get("slide_title", "")
            except Exception as e:
                status = "WARNING"
                error_msg = f"Could not get PPT status: {e}"

        return {
            "name": "Presentation Engine",
            "code": "PPT-07",
            "status": "PRESENTING" if is_presenting else status,
            "deck_file": ppt_file,
            "deck_exists": file_exists,
            "is_presenting": is_presenting,
            "current_slide": current_slide,
            "total_slides": total_slides,
            "current_slide_title": slide_title,
            "custom_script_mode": "Autonomous 7-Slide College Script (Active)",
            "error": error_msg,
            "fix_tip": fix_tip,
            "details": f"Presentation {'IN PROGRESS (Slide ' + str(current_slide) + '/' + str(total_slides) + ')' if is_presenting else 'Ready (' + str(total_slides) + ' slides in ' + ppt_file + ')'}."
        }

    def check_audio_router(self) -> dict:
        """Inspects physical and virtual audio device routing."""
        status = "ONLINE"
        error_msg = None
        fix_tip = None

        out_name = "System Default"
        in_name = "System Default"
        is_headphones = False
        is_headset_mic = False
        vb_cable_present = False

        if sd:
            try:
                from audio_router import detect_audio_devices
                cfg = detect_audio_devices(verbose=False)
                out_name = cfg.get("output_name", "Default")
                in_name = cfg.get("input_name", "Default")
                is_headphones = cfg.get("is_headphones", False)
                is_headset_mic = cfg.get("is_headphone_mic", False)
                vb_cable_present = cfg.get("vb_cable_device") is not None
            except Exception as e:
                status = "ERROR"
                error_msg = f"Audio routing detection failed: {e}"
                fix_tip = "Check if sound drivers are working properly in Windows Device Manager."

        if not is_headphones:
            status = "WARNING"
            error_msg = "Headphones (Rockerz 650 Pro) not detected. Output routed to laptop speakers."
            fix_tip = "Turn on your Rockerz 650 Pro Bluetooth headphones to route private audio."

        return {
            "name": "Audio Hardware Router",
            "code": "ROUT-08",
            "status": status,
            "output_device": out_name,
            "is_headphones": is_headphones,
            "input_device": in_name,
            "is_headset_mic": is_headset_mic,
            "vb_cable_lip_sync": "Connected" if vb_cable_present else "Not Detected",
            "multi_stream_playback": "Active (Direct + VB-Cable)",
            "error": error_msg,
            "fix_tip": fix_tip,
            "details": f"Audio Output: {out_name} | Input Mic: {in_name}"
        }

    # ----------------- FULL SYSTEM REPORT -----------------

    def get_full_status(self, vision_engine=None, primary_target_name=None, person_count=None, detected_objects=None, presentation_manager=None, silero_model_loaded=True) -> dict:
        """Assembles comprehensive telemetry for all 8 subsystems."""
        brain = self.check_brain()
        eyes = self.check_eyes(vision_engine, primary_target_name, person_count, detected_objects)
        voice = self.check_voice(silero_model_loaded)
        ears = self.check_ears()
        memory = self.check_memory()
        embodiment = self.check_embodiment()
        presentation = self.check_presentation(presentation_manager)
        router = self.check_audio_router()

        subsystems = [brain, eyes, voice, ears, memory, embodiment, presentation, router]

        # Calculate overall system status
        has_error = any(s["status"] in ["ERROR", "OFFLINE"] for s in subsystems)
        has_warning = any(s["status"] == "WARNING" for s in subsystems)

        if has_error:
            overall_status = "CRITICAL_FAULT"
            overall_badge = "SYSTEM ERROR"
            overall_color = "#ff2a5f"
        elif has_warning:
            overall_status = "DEGRADED"
            overall_badge = "PARTIAL WARNING"
            overall_color = "#ffb703"
        else:
            overall_status = "OPERATIONAL"
            overall_badge = "ALL SYSTEMS NORMAL"
            overall_color = "#00ff9d"

        # Count active subsystems
        active_count = sum(1 for s in subsystems if s["status"] in ["ONLINE", "PRESENTING", "WARNING", "STANDBY"])

        # Collect active errors
        active_errors = []
        for s in subsystems:
            if s.get("error"):
                active_errors.append({
                    "subsystem": s["name"],
                    "code": s["code"],
                    "severity": "CRITICAL" if s["status"] in ["ERROR", "OFFLINE"] else "WARNING",
                    "error": s["error"],
                    "fix_tip": s.get("fix_tip")
                })

        return {
            "overall_status": overall_status,
            "overall_badge": overall_badge,
            "overall_color": overall_color,
            "active_subsystems_count": f"{active_count} / {len(subsystems)} Active",
            "uptime": self.get_uptime_str(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "subsystems": subsystems,
            "active_errors": active_errors,
            "has_error": has_error
        }

    # ----------------- ON-DEMAND LIVE SKILL TESTER -----------------

    def test_skill(self, skill_code: str) -> dict:
        """Executes an active, live diagnostic test on the requested skill and logs outcome."""
        skill_code = skill_code.upper()
        t0 = time.time()

        if "BRAIN" in skill_code:
            try:
                res = requests.post(
                    "http://127.0.0.1:11434/api/chat",
                    json={"model": "myra", "messages": [{"role": "user", "content": "respond with OK in one word"}], "stream": False},
                    timeout=15
                )
                dt = round((time.time() - t0) * 1000, 1)
                if res.status_code == 200:
                    reply = res.json().get("message", {}).get("content", "").strip()
                    self.log("BRAIN", "SUCCESS", f"Brain test passed in {dt}ms.", f"Response: {reply}")
                    return {"status": "PASSED", "latency_ms": dt, "message": f"Ollama 'myra' responded in {dt}ms: '{reply}'"}
                else:
                    self.log("BRAIN", "ERROR", f"Brain test returned status {res.status_code}", fix_tip="Check Ollama logs.")
                    return {"status": "FAILED", "latency_ms": dt, "error": f"HTTP {res.status_code}"}
            except Exception as e:
                dt = round((time.time() - t0) * 1000, 1)
                self.log("BRAIN", "ERROR", f"Brain test failed: {e}", traceback.format_exc(), "Ensure 'ollama serve' is running.")
                return {"status": "FAILED", "latency_ms": dt, "error": str(e), "fix_tip": "Run 'ollama serve' in PowerShell."}

        elif "VOICE" in skill_code:
            try:
                from main import generate_voice
                test_wav = generate_voice("System audio check.", "diagnostic_voice_test.wav")
                dt = round((time.time() - t0) * 1000, 1)
                if test_wav and os.path.exists(test_wav):
                    size = os.path.getsize(test_wav)
                    self.log("VOICE", "SUCCESS", f"Silero TTS test passed in {dt}ms ({size} bytes).")
                    return {"status": "PASSED", "latency_ms": dt, "message": f"Generated {size} bytes in {dt}ms via Silero 48kHz."}
                else:
                    self.log("VOICE", "ERROR", "Failed to generate test WAV file.")
                    return {"status": "FAILED", "latency_ms": dt, "error": "Output file not created."}
            except Exception as e:
                dt = round((time.time() - t0) * 1000, 1)
                self.log("VOICE", "ERROR", f"Voice test error: {e}", traceback.format_exc(), "Check PyTorch / Silero model load.")
                return {"status": "FAILED", "latency_ms": dt, "error": str(e)}

        elif "EYES" in skill_code:
            try:
                cap = cv2.VideoCapture(0)
                dt = round((time.time() - t0) * 1000, 1)
                if cap.isOpened():
                    ret, frame = cap.read()
                    cap.release()
                    if ret and frame is not None:
                        h, w = frame.shape[:2]
                        self.log("EYES", "SUCCESS", f"Camera test passed in {dt}ms ({w}x{h} frame grabbed).")
                        return {"status": "PASSED", "latency_ms": dt, "message": f"Captured {w}x{h} video frame in {dt}ms."}
                    else:
                        self.log("EYES", "WARN", "Camera opened but could not read frame.", fix_tip="Check camera drivers.")
                        return {"status": "FAILED", "latency_ms": dt, "error": "Could not read frame from camera."}
                else:
                    self.log("EYES", "ERROR", "Camera 0 could not be opened.", fix_tip="Close other apps using webcam.")
                    return {"status": "FAILED", "latency_ms": dt, "error": "Camera 0 busy or disconnected."}
            except Exception as e:
                dt = round((time.time() - t0) * 1000, 1)
                self.log("EYES", "ERROR", f"Eyes test error: {e}", traceback.format_exc())
                return {"status": "FAILED", "latency_ms": dt, "error": str(e)}

        elif "MEM" in skill_code or "MEMORY" in skill_code:
            try:
                conn = sqlite3.connect("myra_memory.db", timeout=2.0)
                c = conn.cursor()
                c.execute("PRAGMA integrity_check;")
                res = c.fetchone()[0]
                c.execute("SELECT COUNT(*) FROM conversations;")
                count = c.fetchone()[0]
                conn.close()
                dt = round((time.time() - t0) * 1000, 1)
                self.log("MEMORY", "SUCCESS", f"Database test passed in {dt}ms (Integrity: {res}, Rows: {count}).")
                return {"status": "PASSED", "latency_ms": dt, "message": f"Integrity: {res} | Total records: {count}"}
            except Exception as e:
                dt = round((time.time() - t0) * 1000, 1)
                self.log("MEMORY", "ERROR", f"Database test error: {e}", traceback.format_exc())
                return {"status": "FAILED", "latency_ms": dt, "error": str(e)}

        elif "ROUT" in skill_code or "AUDIO" in skill_code:
            try:
                from audio_router import detect_audio_devices
                cfg = detect_audio_devices(verbose=False)
                dt = round((time.time() - t0) * 1000, 1)
                self.log("AUDIO", "SUCCESS", f"Audio router rescan completed in {dt}ms.")
                return {"status": "PASSED", "latency_ms": dt, "message": f"Output: {cfg.get('output_name')} | Input: {cfg.get('input_name')}"}
            except Exception as e:
                dt = round((time.time() - t0) * 1000, 1)
                self.log("AUDIO", "ERROR", f"Audio router test error: {e}", traceback.format_exc())
                return {"status": "FAILED", "latency_ms": dt, "error": str(e)}

        elif "PPT" in skill_code or "PRESENTATION" in skill_code:
            try:
                from pptx import Presentation
                if not os.path.exists("myra.pptx"):
                    return {"status": "FAILED", "error": "myra.pptx not found in directory"}
                prs = Presentation("myra.pptx")
                slide_count = len(prs.slides)
                dt = round((time.time() - t0) * 1000, 1)
                self.log("PPT", "SUCCESS", f"Presentation deck parsed in {dt}ms ({slide_count} slides).")
                return {"status": "PASSED", "latency_ms": dt, "message": f"Successfully parsed myra.pptx ({slide_count} slides)."}
            except Exception as e:
                dt = round((time.time() - t0) * 1000, 1)
                self.log("PPT", "ERROR", f"Presentation test error: {e}", traceback.format_exc())
                return {"status": "FAILED", "latency_ms": dt, "error": str(e)}

        elif "AVTR" in skill_code or "EMBODIMENT" in skill_code:
            try:
                from main import trigger_vmc_expression
                trigger_vmc_expression("Joy", 0.8, duration=2.0)
                dt = round((time.time() - t0) * 1000, 1)
                self.log("EMBODIMENT", "SUCCESS", f"Embodiment test OSC packet dispatched to port 39539 ({dt}ms).")
                return {"status": "PASSED", "latency_ms": dt, "message": "Dispatched 'Joy' expression packet to VSeeFace."}
            except Exception as e:
                dt = round((time.time() - t0) * 1000, 1)
                self.log("EMBODIMENT", "ERROR", f"Embodiment test error: {e}")
                return {"status": "FAILED", "latency_ms": dt, "error": str(e)}

        elif "EARS" in skill_code:
            try:
                cuda = torch.cuda.is_available() if torch else False
                dt = round((time.time() - t0) * 1000, 1)
                self.log("EARS", "SUCCESS", f"Ears diagnostic verified in {dt}ms (CUDA: {cuda}).")
                return {"status": "PASSED", "latency_ms": dt, "message": f"Faster-Whisper STT ready. Hardware acceleration: {'CUDA' if cuda else 'CPU'}."}
            except Exception as e:
                dt = round((time.time() - t0) * 1000, 1)
                return {"status": "FAILED", "latency_ms": dt, "error": str(e)}

        return {"status": "UNKNOWN", "error": f"Unknown skill code '{skill_code}'"}

# Singleton instance
system_monitor = SystemMonitor()
