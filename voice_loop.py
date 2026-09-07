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

SAMPLE_RATE = 16000        # Native Whisper sampling rate
THRESHOLD = 0.008          # RMS speech detection volume threshold
SILENCE_LIMIT = 1.2        # Seconds of silence after speech to finish utterance
MAX_RECORD_TIME = 8.0      # Hard maximum recording duration (prevents infinite loop/getting stuck)
MAX_IDLE_TIME = 15         # Max silence before checking idle
ACTIVE_TIMEOUT = 35        # Seconds of inactivity before sleeping

BACKEND_URL = "http://127.0.0.1:8000/chat"
WAKE_WORDS = [
    "myra", "maira", "myrah", "mira", "maya", "aira", "mera", "mayra",
    "माइरा", "मायरा", "मीरा", "मईरा"
]

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

# -------- CUDA DLL Resolution for Windows --------
import site

def setup_cuda_dlls():
    """Adds nvidia CUDA and cuDNN dll paths to PATH and Windows DLL directories."""
    paths_to_add = []
    site_packages_dirs = site.getsitepackages() if hasattr(site, 'getsitepackages') else []
    if hasattr(site, 'getusersitepackages'):
        site_packages_dirs.append(site.getusersitepackages())

    for base in site_packages_dirs:
        nvidia_dir = os.path.join(base, "nvidia")
        if os.path.exists(nvidia_dir):
            for root, dirs, files in os.walk(nvidia_dir):
                if any(f.endswith(".dll") for f in files):
                    paths_to_add.append(root)

        torch_lib = os.path.join(base, "torch", "lib")
        if os.path.exists(torch_lib):
            paths_to_add.append(torch_lib)

    for p in set(paths_to_add):
        if p not in os.environ.get("PATH", ""):
            os.environ["PATH"] = p + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory") and os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass

# -------- Load Whisper STT --------

print("--- [EARS] Initializing Faster-Whisper Model... ---")
setup_cuda_dlls()

whisper_model = None
if torch.cuda.is_available():
    try:
        print(f"--> [EARS] Attempting CUDA Acceleration on: {torch.cuda.get_device_name(0)}")
        whisper_model = WhisperModel("medium", device="cuda", compute_type="float16")
        # Verify CUDA actually works with a 0.1s dummy tensor
        dummy_audio = np.zeros(1600, dtype=np.float32)
        _ = list(whisper_model.transcribe(dummy_audio, beam_size=1)[0])
        print("--> [EARS] Success! CUDA Acceleration is fully active.")
    except Exception as cuda_err:
        print(f"--> [EARS] CUDA link notice ({cuda_err}). Falling back to CPU (int8)...")
        whisper_model = None

if whisper_model is None:
    print("--> [EARS] Running on CPU (int8) mode...")
    whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

print("--- [EARS] Faster-Whisper Ready! ---")

# -------- Helper Functions --------

def wake_detected(text):
    text = text.lower().strip()
    words = text.split()
    for word in words:
        for wake in WAKE_WORDS:
            if difflib.SequenceMatcher(None, wake, word).ratio() > 0.7 or wake in word:
                return True
    return False

import soundfile as sf
import threading

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

def safe_play(file_path):
    """Plays audio through the lip-sync device (VB-Cable / Speakers)."""
    global is_speaking
    if not file_path or not os.path.exists(file_path):
        return

    is_speaking = True
    try:
        data, fs = sf.read(file_path, dtype='float32')
        target_dev = get_playback_device()
        sd.play(data, fs, device=target_dev)
        sd.wait()
    except Exception as e:
        print(f"Audio Playback Warning: {e}, falling back to winsound...")
        try:
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
        except Exception:
            pass
    finally:
        is_speaking = False
        time.sleep(0.4)

def record_audio_in_memory():
    """Captures audio directly into a numpy buffer in RAM with live terminal feedback."""
    global is_speaking
    if is_speaking:
        return None

    audio_chunks = []
    silence_start = None
    speech_started = False
    speech_start_time = None
    idle_start = time.time()

    if is_active:
        print("\r🎧 [LISTENING] Myra is listening to you... (Speak now)      ", end="", flush=True)
    else:
        print("\r💤 [STANDBY] Say 'Myra' to wake her up...                  ", end="", flush=True)

    try:
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
                rms = np.sqrt(np.mean(data**2))

                # Timeout if no speech begins
                if not speech_started:
                    if time.time() - idle_start > MAX_IDLE_TIME:
                        return None
                    if rms > THRESHOLD:
                        speech_started = True
                        speech_start_time = time.time()
                        silence_start = None
                        print("\r🔴 [RECORDING...] Hearing you speak...                     ", end="", flush=True)
                        audio_chunks.append(data.flatten())
                else:
                    audio_chunks.append(data.flatten())
                    # Hard cap: prevent getting stuck in noisy rooms
                    if time.time() - speech_start_time > MAX_RECORD_TIME:
                        break

                    if rms < THRESHOLD:
                        if silence_start is None:
                            silence_start = time.time()
                        elif time.time() - silence_start > SILENCE_LIMIT:
                            break
                    else:
                        silence_start = None

    except Exception as e:
        print(f"\nMic error: {e}")
        return None

    if not audio_chunks:
        return None

    audio_arr = np.concatenate(audio_chunks)

    # Minimum speech filter (at least 0.5 seconds)
    if len(audio_arr) < SAMPLE_RATE * 0.5:
        return None

    return audio_arr

def transcribe_audio(audio_array):
    """Transcribes in-memory float32 audio buffer with Faster-Whisper."""
    global whisper_model
    print("\r⚡ [TRANSCRIBING...] Processing voice on GPU...             ", end="", flush=True)
    try:
        segments, info = whisper_model.transcribe(
            audio_array,
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        text = " ".join([segment.text for segment in segments]).strip()
        print("\r" + " " * 65 + "\r", end="", flush=True)
        return text
    except Exception as e:
        print(f"\nTranscription Notice ({e}), switching to CPU fallback...")
        try:
            whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")
            segments, info = whisper_model.transcribe(
                audio_array,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            text = " ".join([segment.text for segment in segments]).strip()
            print("\r" + " " * 65 + "\r", end="", flush=True)
            return text
        except Exception as cpu_err:
            print(f"\nTranscription Error: {cpu_err}")
            return ""

def send_to_myra(text):
    """Sends text to Myra FastAPI backend."""
    print("🧠 [MYRA THINKING...] Formulating witty response...", end="", flush=True)
    try:
        response = requests.post(BACKEND_URL, json={"message": text}, timeout=60)
        data = response.json()
        print("\r" + " " * 65 + "\r", end="", flush=True)
        return data.get("response"), data.get("audio_file")
    except Exception as e:
        print(f"\nBackend Connection Error: {e}")
        return None, None

def speak_direct(text, output_file="exit.wav"):
    """Synthesizes and plays a direct voice line using Myra's original Silero voice."""
    try:
        res = requests.post("http://127.0.0.1:8000/tts", json={"text": text, "output_file": output_file}, timeout=10)
        data = res.json()
        audio_file = data.get("audio_file")
        if audio_file and os.path.exists(audio_file):
            safe_play(audio_file)
    except Exception as e:
        print(f"\nDirect Speech Notice: {e}")

if __name__ == "__main__":
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