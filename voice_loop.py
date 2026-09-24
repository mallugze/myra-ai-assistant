import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import time
import random
import difflib
import re
import requests
import winsound
import numpy as np
import sounddevice as sd
from collections import deque
from faster_whisper import WhisperModel
import torch
import edge_tts
import asyncio

# -------- Settings --------

SAMPLE_RATE = 16000        # Native Whisper sampling rate
BASE_THRESHOLD = 0.0040    # High-sensitivity RMS speech detection volume threshold (lowered from 0.008)
SILENCE_LIMIT = 1.0        # Seconds of silence after speech to finish utterance
MAX_RECORD_TIME = 8.0      # Hard maximum recording duration (prevents infinite loop/getting stuck)
MAX_IDLE_TIME = 15         # Max silence before checking idle
ACTIVE_TIMEOUT = 35        # Seconds of inactivity before sleeping

BACKEND_URL = "http://127.0.0.1:8000/chat"
WAKE_WORDS = [
    "myra", "maira", "myrah", "mira", "maya", "aira", "mera", "mayra",
    "mahira", "mahir", "meera", "moira", "vaira", "naira", "mora", "mara",
    "mya", "mayrah", "meyra",
    # Hindi spellings
    "माइरा", "मायरा", "मीरा", "मईरा", "माहिरा"
]

WHISPER_PROMPT = "Myra, Mallu, PPT, slide, presentation, Mahira, hey Myra, start presentation, next slide."

WHISPER_HALLUCINATIONS = {
    "you", "you.", "bye", "bye.", "thank you", "thank you.", "i'm sorry", "i'm sorry.",
    "subtitles by", "watching!", "thank you for watching", "see you next time",
    "thank you for watching!", "thanks for watching", "so", "so.", "oh", "oh."
}

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
        cap = torch.cuda.get_device_capability(0)
        # CTranslate2 on current build supports up to sm_90. sm_120 (RTX 50-series) uses CPU fallback
        if cap[0] <= 9:
            print(f"--> [EARS] Attempting CUDA Acceleration on: {torch.cuda.get_device_name(0)}")
            whisper_model = WhisperModel("base", device="cuda", compute_type="float16")
            dummy_audio = np.zeros(1600, dtype=np.float32)
            _ = list(whisper_model.transcribe(dummy_audio, beam_size=1)[0])
            print("--> [EARS] Success! CUDA Acceleration is fully active.")
        else:
            print(f"--> [EARS] RTX 50-series (sm_{cap[0]}{cap[1]}) detected; using high-speed multi-threaded CPU mode...")
    except Exception as cuda_err:
        print(f"--> [EARS] CUDA link notice ({cuda_err}). Falling back to CPU (int8)...")
        whisper_model = None

if whisper_model is None:
    print("--> [EARS] Running on CPU (int8) mode with multi-threading...")
    whisper_model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=8)

print("--- [EARS] Faster-Whisper Ready! ---")

# -------- Helper Functions --------

def correct_domain_phonetics(text):
    """Corrects common Whisper phonetic misrecognitions for Myra's domain."""
    corrections = {
        r'\bpbt\b': 'PPT',
        r'\bbpt\b': 'PPT',
        r'\bppd\b': 'PPT',
        r'\bpp t\b': 'PPT',
        r'\bp\.p\.t\b': 'PPT',
        r'\bpresentation by task\b': 'presentation',
        r'\bmahir\b': 'Myra',
        r'\bmahira\b': 'Myra',
        r'\bmaira\b': 'Myra',
        r'\bvaira\b': 'Myra',
        r'\bmira\b': 'Myra',
        r'\baira\b': 'Myra',
    }
    corrected = text
    for pattern, repl in corrections.items():
        corrected = re.sub(pattern, repl, corrected, flags=re.IGNORECASE)
    return corrected

def wake_detected(text):
    text = text.lower().strip()
    clean_text = re.sub(r'[^\w\s]', '', text)
    words = clean_text.split()

    # 1. Direct word or substring check
    for wake in WAKE_WORDS:
        if wake in text or wake in clean_text:
            return True

    # 2. Fuzzy match against each spoken word (0.65 threshold to catch accent nuances)
    for word in words:
        for wake in WAKE_WORDS:
            if difflib.SequenceMatcher(None, wake, word).ratio() >= 0.65:
                return True

    # 3. Direct presentation or wake intent phrases (wakes her up even if wake word was muffled)
    wake_intent_phrases = [
        "present the ppt", "present ppt", "start presentation", "start the ppt",
        "open the ppt", "show the ppt", "show presentation", "can you present",
        "wake up", "are you there", "are you listening", "hello myra", "hey myra"
    ]
    for phrase in wake_intent_phrases:
        if phrase in clean_text:
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

recent_myra_phrases = []  # list of (clean_text, timestamp)

def add_myra_speech_history(text):
    """Tracks Myra's recent spoken phrases with timestamps for acoustic echo cancellation."""
    global recent_myra_phrases
    if not text:
        return
    clean = re.sub(r"[^\w\s]", "", text.lower()).strip()
    if clean:
        now = time.time()
        recent_myra_phrases = [(p, t) for p, t in recent_myra_phrases if now - t < 12.0]
        recent_myra_phrases.append((clean, now))

def is_echo_of_myra(user_text):
    """Detects if microphone picked up Myra's own voice from the speakers."""
    if not user_text:
        return False

    clean_user = re.sub(r"[^\w\s]", "", user_text.lower()).strip()
    if not clean_user or len(clean_user) < 6:
        return False

    # Never filter out user commands, presentation keywords, or greetings!
    user_words = set(clean_user.split())
    protected_command_words = {
        "ppt", "ppd", "pdf", "presentation", "slide", "slides", "next", "stop",
        "present", "explain", "who", "meet", "how", "what", "where", "myra", "class", "day"
    }
    if user_words.intersection(protected_command_words):
        return False

    now = time.time()
    for phrase, timestamp in recent_myra_phrases:
        # Acoustic echo can only occur while audio is playing or within 3.5s after
        if now - timestamp > 3.5:
            continue

        # Strict verbatim similarity threshold (0.80+) to prevent blocking user's own speech
        ratio = difflib.SequenceMatcher(None, clean_user, phrase).ratio()
        if ratio >= 0.80:
            return True

        if len(clean_user) > 15 and clean_user in phrase:
            return True

    return False

def is_backend_speaking():
    """Checks if either voice loop or backend is actively playing speech."""
    if is_speaking:
        return True
    info = check_backend_status(fast=True)
    if info.get("last_speech"):
        add_myra_speech_history(info.get("last_speech"))
    return info.get("is_proactive_speaking", False) or info.get("is_audio_playing", False)

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
        time.sleep(0.5)

def record_audio_in_memory():
    """
    Captures audio directly into a numpy buffer in RAM with pre-roll buffering
    and automatic gain control.
    """
    if is_backend_speaking():
        return None

    audio_chunks = []
    # Pre-roll ring buffer holds ~450ms (7 chunks * 1024 samples = 7168 samples = 448ms)
    # This completely eliminates cut-off on soft initial consonants like "M" in "Myra".
    preroll_buffer = deque(maxlen=7)

    silence_start = None
    speech_started = False
    speech_start_time = None
    idle_start = time.time()
    chunk_counter = 0

    ambient_measurements = []
    effective_threshold = BASE_THRESHOLD

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
                chunk_counter += 1
                if chunk_counter % 4 == 0:  # Check speaking status every ~100ms
                    if is_backend_speaking():
                        return None

                data, _ = stream.read(1024)
                rms = np.sqrt(np.mean(data**2))

                # Dynamic noise floor adaptation during the first few idle chunks
                if not speech_started and len(ambient_measurements) < 6:
                    ambient_measurements.append(rms)
                    if len(ambient_measurements) == 6:
                        ambient_avg = float(np.mean(ambient_measurements))
                        # Keep threshold sensitive but safely above ambient room noise
                        effective_threshold = max(BASE_THRESHOLD, min(0.0080, ambient_avg * 2.2))

                # Speech trigger
                if not speech_started:
                    preroll_buffer.append(data.flatten())
                    if time.time() - idle_start > MAX_IDLE_TIME:
                        return None
                    if rms > effective_threshold:
                        speech_started = True
                        speech_start_time = time.time()
                        silence_start = None
                        print("\r🔴 [RECORDING...] Hearing you speak...                     ", end="", flush=True)
                        # Prepend the pre-roll chunks so the initial "M-" consonant is never lost!
                        audio_chunks.extend(preroll_buffer)
                else:
                    audio_chunks.append(data.flatten())
                    # Hard cap: prevent getting stuck in noisy rooms
                    if time.time() - speech_start_time > MAX_RECORD_TIME:
                        break

                    if rms < effective_threshold:
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

    # Minimum speech filter (lowered from 0.5s to 0.35s so quick wake words like 'Myra' aren't dropped)
    if len(audio_arr) < int(SAMPLE_RATE * 0.35):
        return None

    # Automatic Gain Control / Peak Normalization:
    # Scales quiet/muffled mic input up to 0.95 peak amplitude so Whisper gets crystal-clear log-mel features
    peak_val = np.max(np.abs(audio_arr))
    if peak_val > 0.002:
        audio_arr = (audio_arr / peak_val) * 0.95

    return audio_arr

def transcribe_audio(audio_array):
    """Transcribes in-memory float32 audio buffer with Faster-Whisper using vocabulary prompt conditioning."""
    global whisper_model
    print("\r⚡ [TRANSCRIBING...] Processing voice...                     ", end="", flush=True)
    try:
        segments, info = whisper_model.transcribe(
            audio_array,
            beam_size=2,
            language="en",
            initial_prompt=WHISPER_PROMPT,
            condition_on_previous_text=False,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=300)
        )
        text = " ".join([segment.text for segment in segments]).strip()
        print("\r" + " " * 65 + "\r", end="", flush=True)
        return text
    except Exception as e:
        print(f"\nTranscription Notice ({e}), switching to CPU fallback...")
        try:
            whisper_model = WhisperModel("base", device="cpu", compute_type="int8", cpu_threads=8)
            segments, info = whisper_model.transcribe(
                audio_array,
                beam_size=2,
                language="en",
                initial_prompt=WHISPER_PROMPT,
                condition_on_previous_text=False,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300)
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
        return data.get("response"), data.get("audio_file"), data.get("raw_response")
    except Exception as e:
        print(f"\nBackend Connection Error: {e}")
        return None, None, None

def speak_direct(text, output_file="exit.wav"):
    """Synthesizes and plays a direct voice line using Myra's original Silero voice."""
    add_myra_speech_history(text)
    try:
        res = requests.post("http://127.0.0.1:8000/tts", json={"text": text, "output_file": output_file}, timeout=25)
        data = res.json()
        audio_file = data.get("audio_file")
        if audio_file and os.path.exists(audio_file):
            safe_play(audio_file)
    except Exception as e:
        print(f"\nDirect Speech Notice: {e}")

def handle_proactive_event(event):
    """Handles autonomous stranger, crowd, and separation events routed exclusively to voice_loop."""
    if not event:
        return
    ev_type = event.get("event_type")
    reply_text = event.get("text", "")
    audio_file = event.get("audio_file")

    if ev_type == "STRANGER_LINGER":
        print("\n\n👤 [STRANGER DETECTED] An unfamiliar person has appeared near Mallu!")
    elif ev_type == "CROWD_ALERT":
        print("\n\n👥 [CROWD DETECTED] Multiple people have entered the room!")
    elif ev_type == "MALLU_MISSING":
        print("\n\n💔 [MALLU MISSING] Mallu stepped away from the camera!")
    elif ev_type == "MALLU_RETURN":
        print("\n\n✨ [MALLU RETURNED] Welcome back, Mallu!")
    else:
        print(f"\n\n⚡ [PROACTIVE EVENT: {ev_type}]")

    if reply_text:
        add_myra_speech_history(reply_text)
        print(f"🎀 [MYRA]: {reply_text}")

    if audio_file and os.path.exists(audio_file):
        safe_play(audio_file)

_last_status_poll = 0
_cached_status = {}

def check_backend_status(fast=False):
    """Checks backend status for proactive triggers (throttled to avoid log spam)."""
    global _last_status_poll, _cached_status
    now = time.time()
    throttle = 0.15 if fast else 0.5
    if now - _last_status_poll < throttle:
        return _cached_status

    _last_status_poll = now
    try:
        res = requests.get("http://127.0.0.1:8000/status", timeout=0.4)
        if res.status_code == 200:
            data = res.json()
            if data.get("proactive_event"):
                handle_proactive_event(data.get("proactive_event"))
            _cached_status = data
            if _cached_status.get("last_speech"):
                add_myra_speech_history(_cached_status.get("last_speech"))
            return _cached_status
    except Exception:
        pass
    return _cached_status

if __name__ == "__main__":
    print("\n✨ Myra Voice Loop is active! Say 'Myra' to wake her up.\n")

    while True:
        if is_backend_speaking():
            time.sleep(0.3)
            continue

        backend_info = check_backend_status()
        if backend_info.get("awaiting_name_confirmation", False) and not is_active:
            is_active = True
            last_interaction_time = time.time()
            print("\n💫 [STATUS] Myra is waiting for stranger introduction...")

        if backend_info.get("is_presenting", False):
            if not is_active:
                is_active = True
                print("\n📽️ [STATUS] Myra is in Presentation Mode! Hands-free slide controls active.")
            last_interaction_time = time.time()

        audio_buffer = record_audio_in_memory()

        # -------- Inactivity / Idle Timeout --------
        if audio_buffer is None:
            if is_active and (time.time() - last_interaction_time > ACTIVE_TIMEOUT):
                is_active = False
                exit_line = random.choice(EXIT_LINES)
                print(f"\n[Myra]: {exit_line}")
                print("💤 [STATUS] Myra went to sleep (Waiting for wake word)...\n")
                speak_direct(exit_line)
            time.sleep(0.4)
            continue

        # -------- Fast In-Memory Transcription --------
        raw_text = transcribe_audio(audio_buffer)

        if not raw_text:
            continue

        # Prevent repeating exact identical glitches
        if raw_text == last_processed_text:
            continue

        # Acoustic Echo Cancellation: ignore if microphone picked up Myra's speakers
        if is_echo_of_myra(raw_text):
            print(f"\n🛡️ [ECHO CANCELLATION] Filtered out speaker self-hearing: \"{raw_text}\"")
            last_processed_text = raw_text
            continue

        # Hallucination filter in standby: ignore phantom "you", "bye", "thank you"
        if not is_active:
            clean_check = re.sub(r"[^\w\s]", "", raw_text.lower()).strip()
            if clean_check in WHISPER_HALLUCINATIONS or raw_text.lower().strip() in WHISPER_HALLUCINATIONS:
                continue

        # Apply domain phonetic normalization (e.g. PBT -> PPT, Mahira -> Myra)
        user_text = correct_domain_phonetics(raw_text)

        last_processed_text = raw_text
        print(f"\n🗣️ [YOU]: {user_text}")

        # -------- Wake Word Check --------
        if not is_active:
            if wake_detected(user_text):
                is_active = True
                last_interaction_time = time.time()
                print("💫 [STATUS] Myra is listening!")
                cleaned_command = re_command = user_text
                for w in WAKE_WORDS:
                    re_command = re_command.lower().replace(w, "").strip()

                if len(re_command) > 3:
                    reply, reply_audio, raw_reply = send_to_myra(user_text)
                    if reply:
                        add_myra_speech_history(reply)
                        print(f"🎀 [MYRA]: {reply}")
                    if reply_audio:
                        safe_play(reply_audio)
                else:
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
        reply, reply_audio, raw_reply = send_to_myra(user_text)

        if reply:
            add_myra_speech_history(reply)
            print(f"🎀 [MYRA]: {reply}")

        if reply_audio:
            safe_play(reply_audio)

        time.sleep(0.3)