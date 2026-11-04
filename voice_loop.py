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
SILENCE_LIMIT = 0.65       # Snappy silence limit after speech (lowered from 1.0s to eliminate 350ms of dead air)
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
from audio_router import detect_audio_devices, play_audio_dual, get_cached_audio_config

# Intelligently discover and prioritize headphones / headset mic over built-in devices
audio_cfg = detect_audio_devices(verbose=True)

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
    """Plays audio through the prioritized output (Headphones/Speakers) and VB-Cable lip-sync."""
    global is_speaking, audio_cfg
    if not file_path or not os.path.exists(file_path):
        return

    is_speaking = True
    try:
        active_cfg = get_cached_audio_config(max_age_sec=3.0)
        play_audio_dual(file_path, active_cfg)
    except Exception as e:
        print(f"Audio Playback Warning: {e}, falling back to winsound...")
        try:
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
        except Exception:
            pass
    finally:
        is_speaking = False
        time.sleep(0.15)

def record_audio_in_memory():
    """
    Captures audio directly into a numpy buffer in RAM with pre-roll buffering
    and automatic gain control, using the prioritized headset/mic device.
    """
    global audio_cfg
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

    in_dev = audio_cfg.get("input_device") if audio_cfg else None
    try:
        stream = sd.InputStream(
            device=in_dev,
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=1024,
            dtype='float32'
        )
    except Exception:
        # Re-detect in case hardware was plugged or unplugged
        audio_cfg = detect_audio_devices(verbose=False)
        in_dev = audio_cfg.get("input_device")
        stream = sd.InputStream(
            device=in_dev,
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=1024,
            dtype='float32'
        )

    try:
        with stream:

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
    # Strictly disallow unprompted stranger and crowd interruptions
    if ev_type in ["STRANGER_LINGER", "CROWD_ALERT"]:
        return

    reply_text = event.get("text", "")
    audio_file = event.get("audio_file")

    if ev_type == "MALLU_MISSING":
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

def is_conversational_about_presentation(text: str) -> bool:
    """
    Detects if the user is having a casual conversation, asking questions,
    discussing feelings, or talking about the presentation in a conversational context,
    rather than giving an actual imperative command to start presenting.
    """
    if not text:
        return False
    text_lower = text.lower().strip()
    clean = re.sub(r'[^\w\s]', ' ', text_lower)
    clean_spaced = f" {clean} "

    # 1. Inquiries about feelings, thoughts, opinions, preparation, review, or feedback
    conversational_phrases = [
        "how did you feel", "how you felt", "how do you feel", "how are you feeling",
        "felt presenting", "felt about", "what did you think", "what do you think",
        "tell me how", "tell us how", "tell me about", "tell us about",
        "are you ready", "are you nervous", "are you excited", "are you confident",
        "did you like", "did you enjoy", "did you present", "did you see",
        "how was", "how were", "how did", "why did you", "why do you",
        "who made", "who helped", "who worked", "who created",
        "explain what", "what is on slide", "what does the slide", "what does your ppt",
        "can you tell me", "can you explain", "could you explain",
        "do you remember", "did we do well", "how do you plan",
        "good job", "great job", "well done", "congratulations", "proud of you",
        "thank you for presenting", "thanks for presenting", "good presentation", "great presentation"
    ]
    for p in conversational_phrases:
        if p in text_lower:
            return True

    # 2. Conversational / temporal / hypothetical / future phrasing
    temporal_markers = [
        " tomorrow ", " yesterday ", " next week ", " in the morning ",
        " later today ", " after class ", " before class ", " big day ",
        " you are going to present ", " you're going to present ",
        " when you present ", " while presenting ", " once you present "
    ]
    for m in temporal_markers:
        if m in clean_spaced:
            return True

    return False

def is_start_presentation_command(text: str) -> bool:
    """
    Determines if the utterance is a genuine imperative command or direct request
    to launch and deliver the presentation right now.
    """
    if not text:
        return False

    # If it is conversational discussion ABOUT the presentation, never start presentation
    if is_conversational_about_presentation(text):
        return False

    text_lower = text.lower().strip()

    # Direct imperative commands
    direct_commands = [
        "start presentation", "start the presentation", "start presenting",
        "start the ppt", "start ppt", "start the ppd", "start ppd",
        "present the ppt", "present ppt", "present the presentation",
        "present our presentation", "present our project", "present the project",
        "present project", "present the slides", "present slides", "present our slides",
        "present the whole ppt", "present whole ppt", "present all slides", "present all the slides",
        "present the deck", "present our deck", "present everything",
        "open the presentation", "open presentation", "open the ppt", "open ppt",
        "open the slides", "open slides", "launch presentation", "launch the presentation",
        "launch the ppt", "launch ppt", "run presentation", "run the presentation", "run the ppt",
        "shuru karo presentation", "presentation start", "presentation shuru karo"
    ]
    for cmd in direct_commands:
        if cmd in text_lower:
            return True

    # Action requests to present now: "can you present the ppt", "can you present our project"
    request_patterns = [
        r'\b(can|could|will|please|would)\s+(you\s+)?(present|start|launch|run)\s+(the\s+|our\s+)?(ppt|ppd|presentation|slides|deck|project)\b',
        r'\b(go\s+ahead\s+and\s+)?(present|start)\s+(the\s+|our\s+)?(ppt|ppd|presentation|slides|deck|project)\b',
        r'\b(present|start)\s+it\s+now\b'
    ]
    for pattern in request_patterns:
        if re.search(pattern, text_lower):
            return True

    return False

def run_voice_auto_presentation(start_slide=1):
    """
    Hands-free autonomous presentation loop for voice_loop.
    Automatically advances through all 8 slides sequentially with natural presenter pauses,
    speaking each slide's script in Myra's original Silero voice.
    """
    global is_active, last_interaction_time
    print("\n🚀 [AUTO-PRESENTATION] Hands-free presentation activated...")
    print("💡 Myra will present all 8 slides and advance Google Slides automatically.")
    print("💡 Press Ctrl+C at any moment to pause.\n")

    try:
        total_slides = 8
        curr_slide = start_slide
        try:
            status_res = requests.get("http://127.0.0.1:8000/status", timeout=1.0)
            if status_res.status_code == 200:
                s_data = status_res.json()
                total_slides = s_data.get("total_slides", 8)
                curr_slide = s_data.get("current_slide", start_slide)
        except Exception:
            pass

        for next_s in range(curr_slide + 1, total_slides + 1):
            time.sleep(1.2)  # Natural presenter pause before advancing
            print(f"\n➡️ [AUTO-ADVANCE] Advancing Google Slides to Slide {next_s} of {total_slides}...")

            try:
                res = requests.post("http://127.0.0.1:8000/present_ppt", json={"action": "next"}, timeout=60)
                if res.status_code != 200:
                    print(f"⚠️ [AUTO-ADVANCE NOTICE] Backend returned status code {res.status_code}")
                    break

                data = res.json()
                speech = data.get("response")
                audio = data.get("audio_file")
                stat = data.get("status", {})

                if speech:
                    add_myra_speech_history(speech)
                    print(f"🎀 [MYRA]: {speech}")

                if audio and os.path.exists(audio):
                    safe_play(audio)

                last_interaction_time = time.time()

                if not stat.get("is_presenting", True):
                    break

            except Exception as slide_err:
                print(f"⚠️ [AUTO-ADVANCE NOTICE] Error presenting Slide {next_s}: {slide_err}")
                break

        # Mark presentation finished on backend so is_presenting resets to False
        try:
            requests.post("http://127.0.0.1:8000/present_ppt", json={"action": "stop"}, timeout=5)
        except Exception:
            pass

        print("\n" + "★" * 65)
        print(f"🎉 [PRESENTATION COMPLETE] All {total_slides} slides delivered flawlessly!")
        print("❓ Floor is open for Q&A with Myra!")
        print("★" * 65 + "\n")

    except KeyboardInterrupt:
        print("\n⏸️ [AUTO-PRESENTATION] Paused by user. Returning to voice standby.\n")
    except Exception as e:
        print(f"\n⚠️ [AUTO-PRESENTATION NOTICE] {e}\n")

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
                    if is_start_presentation_command(user_text):
                        run_voice_auto_presentation()
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

        if is_start_presentation_command(user_text):
            run_voice_auto_presentation()

        time.sleep(0.3)