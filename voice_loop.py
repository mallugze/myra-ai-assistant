import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write
from faster_whisper import WhisperModel
import requests
import os
import time
import random
import winsound
import librosa
import difflib

# -------- Settings --------

SAMPLE_RATE = 48000
THRESHOLD = 0.015
SILENCE_LIMIT = 2.5
MAX_IDLE_TIME = 15
ACTIVE_TIMEOUT = 30

BACKEND_URL = "http://127.0.0.1:8000/chat"

WAKE_WORDS = ["myra", "maira", "myrah", "mira", "maya"]

EXIT_LINES = [
    "Hmm fine. I will pretend I was not waiting.",
    "Going quiet are we? Typical.",
    "Alright then. I will be here. Obviously.",
    "Wow. Abandoned already? Impressive.",
    "Silent treatment? Cute."
]

# -------- State --------

is_active = False
last_interaction_time = time.time()
last_processed_text = ""
is_speaking = False

# -------- Load Whisper --------

print("Loading Whisper model...")
model = WhisperModel("medium", device="cuda", compute_type="float16")
print("Whisper ready.")

# -------- Helper Functions --------

def wake_detected(text):
    text = text.lower()

    for word in WAKE_WORDS:
        similarity = difflib.SequenceMatcher(None, word, text).ratio()
        if similarity > 0.6:
            return True

    return False


def safe_play(file_path):
    global is_speaking

    is_speaking = True

    while not os.path.exists(file_path):
        time.sleep(0.05)

    winsound.PlaySound(file_path, winsound.SND_FILENAME)

    is_speaking = False


def record_until_silence():
    print("\n🎤 Listening...")

    audio_data = []
    silence_start = None
    speech_started = False
    idle_start = time.time()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            device=1,
            channels=1,
            blocksize=1024,
            dtype='float32'
        ) as stream:

            while True:
                data, _ = stream.read(1024)
                volume = np.linalg.norm(data)

                # Idle timeout before speech
                if not speech_started and time.time() - idle_start > MAX_IDLE_TIME:
                    return None

                if volume > THRESHOLD:
                    speech_started = True
                    silence_start = None
                    audio_data.append(data)

                else:
                    if speech_started:
                        if silence_start is None:
                            silence_start = time.time()

                        elif time.time() - silence_start > SILENCE_LIMIT:
                            break

                        audio_data.append(data)

    except Exception as e:
        print("Mic error:", e)
        return None

    if not audio_data:
        return None

    audio = np.concatenate(audio_data)

    # Convert multi-channel → mono
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)

    # Minimum speech duration
    if len(audio) < SAMPLE_RATE * 0.8:
        return None

    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))

    write("temp.wav", SAMPLE_RATE, audio)
    return "temp.wav"


def transcribe(file_path):
    audio, sr = librosa.load(file_path, sr=16000)
    segments, _ = model.transcribe(audio, language="en")

    text = ""
    for segment in segments:
        text += segment.text

    return text.strip()


def send_to_myra(text):
    try:
        response = requests.post(
            BACKEND_URL,
            json={"message": text}
        )
        data = response.json()
        return data.get("response"), data.get("audio_file")

    except:
        return None, None


# -------- Main Loop --------

while True:

    # Block mic while speaking
    if is_speaking:
        time.sleep(0.2)
        continue

    audio_file = record_until_silence()

    # -------- Idle Handling --------
    if audio_file is None:
        if is_active and time.time() - last_interaction_time > ACTIVE_TIMEOUT:
            is_active = False

            exit_line = random.choice(EXIT_LINES)
            print(f"Myra: {exit_line}")

            # Generate voice via backend
            reply, reply_audio = send_to_myra(exit_line)

            if reply_audio:
                safe_play(reply_audio)

            print("💤 Deactivated.")

        time.sleep(2)
        continue

    # -------- Transcription --------
    user_text = transcribe(audio_file)

    if not user_text:
        continue

    # Prevent duplicate loop
    if user_text == last_processed_text:
        continue

    last_processed_text = user_text

    print(f"\nYou said: {user_text}")

    # -------- Wake Word Logic --------
    if not is_active:
        if wake_detected(user_text):
            is_active = True
            last_interaction_time = time.time()
            print("💫 Activated.")
        continue

    # -------- Active Mode --------
    last_interaction_time = time.time()

    reply, reply_audio = send_to_myra(user_text)

    if reply:
        print(f"Myra: {reply}")

    if reply_audio:
        safe_play(reply_audio)

    time.sleep(0.5)