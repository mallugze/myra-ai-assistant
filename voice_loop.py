import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import time
import random
import difflib
import requests
import winsound
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
import torch
import edge_tts
import asyncio

# -------- Settings --------

SAMPLE_RATE = 16000        # Native Whisper sampling rate (eliminates resampling lag)
THRESHOLD = 0.02           # Speech detection volume threshold
SILENCE_LIMIT = 1.8        # Seconds of silence before finalizing sentence
MAX_IDLE_TIME = 15         # Max silence before checking idle
ACTIVE_TIMEOUT = 35        # Seconds of inactivity before sleeping

BACKEND_URL = "http://127.0.0.1:8000/chat"
WAKE_WORDS = ["myra", "maira", "myrah", "mira", "maya", "aira", "mera"]

EXIT_LINES = [
    "Hmm, fine. I will pretend I was not waiting anyway.",
    "Going quiet are we? Typical Mallu.",
    "Alright then, I will be right here on your screen.",
    "Wow. Abandoned already? Cute.",
    "Silent treatment? As if you can resist talking to me for long."
]

# -------- State --------

is_active = False
last_interaction_time = time.time()
last_processed_text = ""
is_speaking = False

# -------- Load Whisper STT --------

print("--- [EARS] Loading Faster-Whisper Model... ---")
if torch.cuda.is_available():
    print(f"--> [EARS] CUDA GPU Detected: {torch.cuda.get_device_name(0)}")
    whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")
else:
    print("--> [EARS] Running on CPU (int8)...")
    whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

print("--- [EARS] Faster-Whisper Ready! ---")

# -------- Helper Functions --------

def wake_detected(text):
    text = text.lower().strip()
    words = text.split()
    for word in words:
        for wake in WAKE_WORDS:
            if difflib.SequenceMatcher(None, wake, word).ratio() > 0.7:
                return True
    return False

def safe_play(file_path):
    """Plays audio and sets is_speaking flag to prevent hearing own voice."""
    global is_speaking
    if not file_path or not os.path.exists(file_path):
        return

    is_speaking = True
    try:
        winsound.PlaySound(file_path, winsound.SND_FILENAME)
    except Exception as e:
        print(f"Audio Playback Notice: {e}")
    finally:
        is_speaking = False
        time.sleep(0.3)

def record_audio_in_memory():
    """Captures audio directly into a numpy buffer in RAM (Zero Disk I/O)."""
    global is_speaking
    if is_speaking:
        return None

    audio_chunks = []
    silence_start = None
    speech_started = False
    idle_start = time.time()

    try:
        # Auto-select default input device
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=1024,
            dtype='float32'
        ) as stream:

            while True:
                if is_speaking:
                    return None

                data, _ = stream.read(1024)
                volume = np.linalg.norm(data)

                # Timeout if no speech begins
                if not speech_started:
                    if time.time() - idle_start > MAX_IDLE_TIME:
                        return None

                if volume > THRESHOLD:
                    speech_started = True
                    silence_start = None
                    audio_chunks.append(data.flatten())
                else:
                    if speech_started:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > SILENCE_LIMIT:
                            break
                        audio_chunks.append(data.flatten())

    except Exception as e:
        print(f"Mic error: {e}")
        return None

    if not audio_chunks:
        return None

    audio_arr = np.concatenate(audio_chunks)

    # Minimum speech filter (at least 0.6 seconds)
    if len(audio_arr) < SAMPLE_RATE * 0.6:
        return None

    return audio_arr

def transcribe_audio(audio_array):
    """Transcribes in-memory float32 audio buffer with Faster-Whisper."""
    try:
        # Transcribe directly from memory buffer without saving to disk
        segments, info = whisper_model.transcribe(
            audio_array,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        text = " ".join([segment.text for segment in segments]).strip()
        return text
    except Exception as e:
        print(f"Transcription Error: {e}")
        return ""

def send_to_myra(text):
    """Sends text to Myra FastAPI backend."""
    try:
        response = requests.post(BACKEND_URL, json={"message": text}, timeout=20)
        data = response.json()
        return data.get("response"), data.get("audio_file")
    except Exception as e:
        print(f"Backend Connection Error: {e}")
        return None, None

def speak_direct(text, output_file="exit.wav"):
    """Synthesizes and plays a direct voice line (for exit/idle lines)."""
    try:
        async def run():
            comm = edge_tts.Communicate(text, "en-US-AnaNeural")
            await comm.save(output_file)
        asyncio.run(run())
        safe_play(output_file)
    except Exception as e:
        print(f"Direct Speech Notice: {e}")

# -------- Main Voice Loop --------

print("\n✨ Myra Voice Loop is active! Say 'Myra' to wake her up.\n")

while True:
    if is_speaking:
        time.sleep(0.1)
        continue

    audio_buffer = record_audio_in_memory()

    # -------- Inactivity / Idle Timeout --------
    if audio_buffer is None:
        if is_active and (time.time() - last_interaction_time > ACTIVE_TIMEOUT):
            is_active = False
            exit_line = random.choice(EXIT_LINES)
            print(f"\n[Myra]: {exit_line}")
            print("💤 [STATUS] Myra went to sleep (Waiting for wake word)...\n")
            speak_direct(exit_line)
        time.sleep(0.5)
        continue

    # -------- Fast In-Memory Transcription --------
    user_text = transcribe_audio(audio_buffer)

    if not user_text:
        continue

    # Prevent repeating exact identical glitches
    if user_text == last_processed_text:
        continue
    last_processed_text = user_text

    print(f"\n🗣️ [YOU]: {user_text}")

    # -------- Wake Word Check --------
    if not is_active:
        if wake_detected(user_text):
            is_active = True
            last_interaction_time = time.time()
            print("💫 [STATUS] Myra is listening!")
            # If user said more than just the wake word (e.g. "Myra what are you doing")
            cleaned_command = re_command = user_text
            for w in WAKE_WORDS:
                re_command = re_command.lower().replace(w, "").strip()
            
            if len(re_command) > 3:
                reply, reply_audio = send_to_myra(user_text)
                if reply:
                    print(f"🎀 [MYRA]: {reply}")
                if reply_audio:
                    safe_play(reply_audio)
            else:
                # Quick playful wake acknowledgment
                wake_replies = [
                    "Yes, Mallu? Miss me already?",
                    "I'm listening. What do you need?",
                    "Here! What's up?",
                    "Did someone call their favorite companion?"
                ]
                ack = random.choice(wake_replies)
                print(f"🎀 [MYRA]: {ack}")
                speak_direct(ack)
        continue

    # -------- Active Conversation Mode --------
    last_interaction_time = time.time()
    reply, reply_audio = send_to_myra(user_text)

    if reply:
        print(f"🎀 [MYRA]: {reply}")

    if reply_audio:
        safe_play(reply_audio)

    time.sleep(0.2)