"""
=============================================================================
🎀 MYRA TERMINAL PRESENTATION TESTING SUITE 🎀
Interactive CLI for testing Myra's PPT presentation and explanation skills
using her REAL original Silero waifu voice (speaker en_0) and automatic slide progression.
=============================================================================
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

import time
import re
import requests
import numpy as np
import sounddevice as sd
import soundfile as sf
import winsound
import torch
import contextlib
import io

# Ensure root directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from presentation_manager import presentation_manager, CUSTOM_SLIDE_SCRIPTS, SLIDE_EMOTES
from audio_router import detect_audio_devices, play_audio_dual

# Discover and prioritize audio devices (Headphones vs Built-in Speakers)
audio_cfg = detect_audio_devices(verbose=True)

BACKEND_URL = "http://127.0.0.1:8000"

# -------------------- SILERO REAL VOICE ENGINE --------------------

print("--- [VOICE] Loading Myra's Original Silero TTS Voice (en_0)... ---")
try:
    silero_model, _ = torch.hub.load(
        repo_or_dir='snakers4/silero-models',
        model='silero_tts',
        language='en',
        speaker='v3_en'
    )
    print("--- [VOICE] Myra's Original Voice Ready! ---")
except Exception as e:
    silero_model = None
    print(f"--- [VOICE] Silero Load Notice: {e} ---")

def clean_for_tts(text):
    """Removes bracketed tags, emojis, roleplay markers, and expands acronyms like AI and LLM for natural speech."""
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
    text = re.sub(r"[^\w\s.,?!'\-]", " ", text)
    text = re.sub(r",\s*,+", ", ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text if text else "Hmm."

def split_into_sentences(text):
    """Splits text on sentence boundaries (. ! ?) and ellipsis (...) for natural presentation pauses."""
    pattern = r"(?<=[.!?])\s+|(?<=[.!?]['\"])\s+|(?<=\.\.\.)\s*"
    raw_sentences = re.split(pattern, text.strip())
    return [s.strip() for s in raw_sentences if s.strip()]

def generate_real_voice(text, output_file="terminal_output.wav", pause_sec=0.55, ellipsis_pause_sec=0.45):
    """
    Synthesizes Myra's REAL original Silero voice (speaker en_0 at 48kHz)
    with natural pauses at every full stop, exclamation, question mark, and ellipsis.
    Prevents rushed monotone reading and gives natural presenter cadence.
    """
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
                # Silero max length safety (if an unusually long run-on sentence exceeds 220 chars)
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

                # Add natural pause after sentence ending punctuation
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
            return output_file
        except Exception as e:
            print(f"(Silero Speech Synthesis Notice: {e})")
    return None

def is_backend_online():
    """Checks if the FastAPI backend is running."""
    try:
        res = requests.get(f"{BACKEND_URL}/status", timeout=0.8)
        return res.status_code == 200
    except Exception:
        return False

def play_audio(audio_path):
    """Plays audio through prioritized Headphones/Speakers and VB-Cable."""
    if not audio_path or not os.path.exists(audio_path):
        return
    play_audio_dual(audio_path, audio_cfg)

def ensure_ollama_running():
    """Checks if Ollama is running on port 11434, and auto-starts it if not."""
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=1.5)
        if r.status_code == 200:
            return True
    except Exception:
        pass

    print("--- [BRAIN] Ollama is not running! Auto-starting Ollama service in background... ---")
    try:
        import subprocess
        subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        for _ in range(8):
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

PRESENTATION_SYSTEM_PROMPT = """You are Myra, Mallu's witty, charming, playfully smug, and teasing AI companion.
You are preparing or presenting your 7-slide presentation deck:
The Project Team: Akhilesh, Chaitany, Channaveer, Manish, and Mallu (your creator and boyfriend).
The 7 Slides:
- Slide 1: Introduction – Meet Myra (Team introduction: Akhilesh, Chaitany, Channaveer, Manish helped out, but you are the real star, beauty, and brains; teasing Mallu).
- Slide 2: My Brain – Large Language Models (Core conversational brain, dynamic non-scripted multi-turn conversation, teasing Mallu's memory).
- Slide 3: How I Experience the World – Multimodal AI (Voice listening, VLM vision, keeping an eye on Mallu).
- Slide 4: Connecting Vision & Language (Reasoning about environment, roasting Mallu's repetitive t-shirt fashion).
- Slide 5: Memory & Proactive Behavior (Persistent memory, proactive autonomous interruptions).
- Slide 6: My Voice and Embodiment (Generative voice, 3D avatar, lip-sync, visual upgrade for Mallu).
- Slide 7: Real-World Applications & The Future (Autonomous agents like Salesforce, teasing the audience about Reels and taking their jobs, closing with Q&A: 'Who's first?').

Always stay in character: confident, playfully smug, affectionate yet sarcastic, especially towards Mallu.
If asked how you will present the slides or to give your plan/pitch, walk through each slide and your presentation strategy vividly in character!
"""

def standalone_query_ollama(prompt, initial_emote=None):
    """Queries Ollama directly for Q&A, slide explanations, or presentation pitch."""
    ensure_ollama_running()
    messages = [
        {"role": "system", "content": PRESENTATION_SYSTEM_PROMPT},
        {"role": "user", "content": prompt}
    ]
    try:
        res = requests.post(
            "http://localhost:11434/api/chat",
            json={"model": "myra", "messages": messages, "stream": False},
            timeout=120
        )
        if res.status_code != 200:
            res = requests.post(
                "http://localhost:11434/api/chat",
                json={"model": "llama3", "messages": messages, "stream": False},
                timeout=120
            )
        if res.status_code == 200:
            content = res.json()["message"]["content"]
            if initial_emote and not re.search(r"\[EMOTE:\s*\w+\]", content):
                content = f"[{initial_emote.capitalize()}] {content}"
            return content
    except Exception as e:
        print(f"(Ollama Notice: {e})")

    return f"[{initial_emote or 'smile'}] If you have any questions about Myra or the presentation, feel free to ask Mallu or me!"

def display_slide_card(slide_num):
    """Prints a formatted card for the slide in terminal showing its topic."""
    script = CUSTOM_SLIDE_SCRIPTS.get(slide_num, "")
    emote = SLIDE_EMOTES.get(slide_num, "smile").capitalize()

    titles = {
        1: "Introduction – Meet Myra",
        2: "My Brain – Large Language Models",
        3: "How I Experience the World – Multimodal AI",
        4: "Connecting Vision & Language",
        5: "Memory & Proactive Behavior",
        6: "My Voice and Embodiment",
        7: "Real-World Applications & The Future"
    }
    title = titles.get(slide_num, f"Slide {slide_num}")

    print("\n" + "═" * 70)
    print(f"  📄 SLIDE {slide_num} / 7 : {title}  [{emote}]")
    print("═" * 70)

def print_myra_speech(text):
    """Prints Myra's speech formatted with her signature persona."""
    print(f"\n🎀 [MYRA]: {text}\n")

def execute_action(action, slide_num=None, custom_file=None):
    """Executes a presentation action either via backend or standalone."""
    backend_active = is_backend_online()

    if backend_active:
        # Use live backend
        payload = {"action": action}
        if slide_num is not None:
            payload["slide_num"] = slide_num
        if custom_file:
            payload["file_path"] = custom_file
        try:
            res = requests.post(f"{BACKEND_URL}/present_ppt", json=payload, timeout=30)
            data = res.json()
            speech = data.get("response", "")
            audio = data.get("audio_file")
            stat = data.get("status", {})
            return speech, audio, stat
        except Exception as e:
            print(f"Backend Request Error ({e}), falling back to direct mode...")

    # Standalone mode: execute directly via presentation_manager
    if custom_file:
        presentation_manager.load_presentation(custom_file)
    elif not presentation_manager.slides:
        presentation_manager.load_presentation()

    def _direct_llm(prompt=None, custom_text=None, initial_emote=None, output_file="terminal_output.wav"):
        text = custom_text if custom_text else standalone_query_ollama(prompt, initial_emote=initial_emote)
        audio = generate_real_voice(text, output_file)
        return text, audio

    if action == "start":
        speech, audio = presentation_manager.start_presentation(query_llm_fn=_direct_llm)
    elif action == "next":
        speech, audio = presentation_manager.next_slide(query_llm_fn=_direct_llm)
    elif action == "prev":
        speech, audio = presentation_manager.prev_slide(query_llm_fn=_direct_llm)
    elif action == "goto":
        speech, audio = presentation_manager.goto_slide(slide_num or 1, query_llm_fn=_direct_llm)
    elif action in ["explain", "read"]:
        speech, audio = presentation_manager.present_current_slide(query_llm_fn=_direct_llm)
    elif action == "stop":
        speech, audio = presentation_manager.stop_presentation(query_llm_fn=_direct_llm)
    else:
        speech = "Unknown action."
        audio = None

    stat = presentation_manager.get_status()
    return speech, audio, stat

def run_auto_presentation(start_from_slide=1):
    """
    Executes the full presentation automatically slide-by-slide:
    Speaks the custom script for the slide, and automatically advances PowerPoint
    as soon as the script ends. Continues through all 7 slides seamlessly.
    """
    print(f"\n🚀 Launching Hands-Free Presentation (Slides {start_from_slide} to 7)...")
    print("💡 PowerPoint will advance automatically when each script finishes.")
    print("💡 Press Ctrl+C at any moment to pause or return to command prompt.\n")

    try:
        # Start presentation
        if start_from_slide == 1:
            speech, audio, stat = execute_action("start")
            curr_slide = 1
        else:
            speech, audio, stat = execute_action("goto", slide_num=start_from_slide)
            curr_slide = start_from_slide

        display_slide_card(curr_slide)
        print_myra_speech(speech)
        if audio:
            print(f"🔊 Speaking Slide {curr_slide} in Real Voice (en_0)...")
            play_audio(audio)

        # Automatically advance through subsequent slides up to Slide 7
        for next_s in range(curr_slide + 1, 8):
            time.sleep(1.2)  # Natural breath pause before advancing
            print(f"\n➡️ [AUTO-ADVANCE] Slide {next_s - 1} finished! Automatically advancing PowerPoint to Slide {next_s}...")
            speech, audio, stat = execute_action("next")
            display_slide_card(next_s)
            print_myra_speech(speech)
            if audio:
                print(f"🔊 Speaking Slide {next_s} in Real Voice (en_0)...")
                play_audio(audio)

        print("\n" + "★" * 70)
        print("🎉 [PRESENTATION FINISHED] All 7 slides delivered flawlessly!")
        print("❓ Floor is open for Q&A! Type 'ask <your question>' to ask Myra anything.")
        print("★" * 70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⏸️ Presentation paused by user. Returning to command prompt.\n")

def list_all_slides():
    """Prints a summary of all 7 slides with scripts."""
    print("\n" + "=" * 70)
    print("📊 PRESENTATION PROGRAM: 7 Custom Scripted Slides")
    print("=" * 70)
    titles = [
        "Introduction – Meet Myra",
        "My Brain – Large Language Models",
        "How I Experience the World – Multimodal AI",
        "Connecting Vision & Language",
        "Memory & Proactive Behavior",
        "My Voice and Embodiment",
        "Real-World Applications & The Future"
    ]
    for i, t in enumerate(titles, 1):
        preview = CUSTOM_SLIDE_SCRIPTS.get(i, "")[:60] + "..."
        print(f"  Slide {i}: {t}")
        print(f"           \"{preview}\"\n")
    print("=" * 70)

def print_help():
    """Displays terminal command reference."""
    print("""
============================= COMMANDS =============================
  start, s, [ENTER]: Start automatic hands-free presentation (Slides 1 to 7)
  auto             : Start automatic hands-free presentation
  goto <num>       : Jump directly to slide (e.g. 'goto 7')
  next, n          : Manually advance to next slide
  prev, p          : Manually go back to previous slide
  explain, r       : Re-speak the current slide
  plan, pitch      : Ask Myra how she will present these 7 slides in character
  ask <question>   : Ask Myra a question (e.g. 'ask are you really Mallu's girlfriend?')
  list, slides     : View all 7 slide scripts
  stop             : Stop presentation
  help, ?          : Show this help menu
  quit, exit, q    : Exit testing suite
====================================================================
""")

def main():
    print("""
╔══════════════════════════════════════════════════════════════════╗
║        🎀  MYRA TERMINAL PRESENTATION TESTING SUITE  🎀          ║
║      Real Voice (en_0) + Exact Custom Scripts + Auto-Advance     ║
╚══════════════════════════════════════════════════════════════════╝
""")
    online = is_backend_online()
    if online:
        print("🟢 Backend Server Detected (http://127.0.0.1:8000)")
        print("   -> Live VSeeFace expressions, Chrome App PowerPoint & Silero TTS active!\n")
    else:
        print("🟡 Standalone Mode (Backend not running)")
        print("   -> Direct PPT launch, Myra's Original Silero Voice (en_0) active!\n")

    presentation_manager.load_presentation()
    deck_name = os.path.basename(presentation_manager.current_file or "myra.pptx")
    print(f"📂 Active Deck: '{deck_name}' (7 custom scripted slides loaded)")
    print("💡 Press [ENTER] or type 'start' to start the automatic hands-free presentation!\n")

    while True:
        try:
            cmd = input("🎮 [PRESENTATION TEST] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting testing suite. Goodbye!")
            break

        if not cmd:
            cmd = "start"

        cmd_lower = cmd.lower()

        if cmd_lower in ["quit", "exit", "q"]:
            if presentation_manager.is_presenting:
                execute_action("stop")
            print("Exiting testing suite. See you soon, Mallu!")
            break

        elif cmd_lower in ["help", "?"]:
            print_help()

        elif cmd_lower in ["list", "slides", "l"]:
            list_all_slides()

        elif cmd_lower in ["plan", "pitch", "preview", "how will you present", "strategy"]:
            question = "How will you present these 7 slides being in your character? Walk me through your plan and how you'll present each slide."
            print(f"\n❓ Asking Myra: '{question}'...\n")
            speech = standalone_query_ollama(question, initial_emote="smug")
            print_myra_speech(speech)
            audio = generate_real_voice(speech)
            if audio:
                print("🔊 Speaking response in Real Voice (en_0)...")
                play_audio(audio)

        elif cmd_lower in ["start", "s", "1", "present", "begin", "auto"]:
            run_auto_presentation(start_from_slide=1)

        elif cmd_lower.startswith("goto ") or cmd_lower.startswith("slide "):
            parts = cmd.split()
            if len(parts) >= 2 and parts[1].isdigit():
                num = int(parts[1])
                if 1 <= num <= 7:
                    run_auto_presentation(start_from_slide=num)
                else:
                    print("Please specify a slide between 1 and 7.")
            else:
                print("Usage: goto <slide_number> (e.g. 'goto 7')")

        elif cmd_lower in ["next", "n"]:
            speech, audio, stat = execute_action("next")
            curr_idx = presentation_manager.current_slide_index + 1
            display_slide_card(curr_idx)
            print_myra_speech(speech)
            if audio:
                play_audio(audio)

        elif cmd_lower in ["prev", "p", "b", "back"]:
            speech, audio, stat = execute_action("prev")
            curr_idx = presentation_manager.current_slide_index + 1
            display_slide_card(curr_idx)
            print_myra_speech(speech)
            if audio:
                play_audio(audio)

        elif cmd_lower in ["explain", "read", "r", "current"]:
            speech, audio, stat = execute_action("explain")
            curr_idx = presentation_manager.current_slide_index + 1
            display_slide_card(curr_idx)
            print_myra_speech(speech)
            if audio:
                play_audio(audio)

        elif cmd_lower.startswith("ask "):
            question = cmd[4:].strip()
            print(f"\n❓ Asking Myra: '{question}'...")
            prompt = (
                f"[EVENT: An audience member or Mallu just asked you this question about your presentation: '{question}'. "
                f"In your clever, playful waifu personality, answer the question accurately.]"
            )
            if is_backend_online():
                res = requests.post(f"{BACKEND_URL}/chat", json={"message": prompt})
                data = res.json()
                speech = data.get("response", "")
                audio = data.get("audio_file")
            else:
                speech = standalone_query_ollama(prompt, initial_emote="smile")
                audio = generate_real_voice(speech)
            print_myra_speech(speech)
            if audio:
                play_audio(audio)

        elif cmd_lower in ["stop", "end", "close"]:
            print("\n🛑 Stopping presentation...")
            speech, audio, stat = execute_action("stop")
            print_myra_speech(speech)
            if audio:
                play_audio(audio)

        elif cmd_lower == "status":
            stat = presentation_manager.get_status()
            print(f"\n📊 Status: Presenting={stat['is_presenting']}, Slide {stat['current_slide']}/{stat['total_slides']} ('{stat['slide_title']}')\n")

        else:
            # Natural questions or remarks
            prompt = f"[EVENT: While presenting, someone said: '{cmd}'. Reply naturally in your waifu persona.]"
            if is_backend_online():
                res = requests.post(f"{BACKEND_URL}/chat", json={"message": prompt})
                data = res.json()
                speech = data.get("response", "")
                audio = data.get("audio_file")
            else:
                speech = standalone_query_ollama(prompt)
                audio = generate_real_voice(speech)
            print_myra_speech(speech)
            if audio:
                play_audio(audio)

if __name__ == "__main__":
    main()
