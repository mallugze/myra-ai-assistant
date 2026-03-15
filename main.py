from fastapi import FastAPI
from pydantic import BaseModel
import requests
import re
import torch
import soundfile as sf
import sqlite3
import time
import contextlib
import io

# -------------------- DATABASE --------------------

conn = sqlite3.connect("myra_memory.db", check_same_thread=False)
cursor = conn.cursor()

# ---- Create tables ----

cursor.execute("""
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    role TEXT,
    content TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS personality_memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    key TEXT UNIQUE,
    value TEXT
)
""")

conn.commit()

# ---- Load previous conversations ----

cursor.execute("""
SELECT role, content FROM conversations
ORDER BY id DESC LIMIT 40
""")

rows = cursor.fetchall()

conversation_history = []
for role, content in reversed(rows):
    conversation_history.append({
        "role": role,
        "content": content
    })

# ---- Load personality memory ----

cursor.execute("SELECT key, value FROM personality_memory")
personality_rows = cursor.fetchall()
personality_facts = {key: value for key, value in personality_rows}

if "relationship_level" not in personality_facts:
    cursor.execute(
        "INSERT OR IGNORE INTO personality_memory (key, value) VALUES (?, ?)",
        ("relationship_level", "1")
    )
    conn.commit()
    personality_facts["relationship_level"] = "1"

print(f"Loaded {len(conversation_history)} messages from memory.")

# -------------------- APP --------------------

app = FastAPI()

# -------------------- TTS MODEL --------------------

model_en, _ = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='en',
    speaker='v3_en'
)

# -------------------- REQUEST MODEL --------------------

class ChatRequest(BaseModel):
    message: str

# -------------------- CLEANING FUNCTION --------------------

def clean_for_tts(text):
    # Remove stage directions like *yawns*, (laughs), [sighs]
    text = re.sub(r"[\*\(\[].*?[\*\)\]]", "", text)

    # Remove unwanted special characters
    text = re.sub(r"[^\w\s.,?!]", "", text)

    # Normalize spacing
    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"(Ahaha(ha)+!?)", "Haha!", text)

    text = text.strip()

    if not text:
        text = "Hmm."

    return text

# -------------------- MEMORY EXTRACTION --------------------

def extract_memory_from_message(user_message):
    try:
        extraction_prompt = [
            {
                "role": "system",
                "content": (
                    "Extract long-term memory facts OR teasing hooks.\n"
                    "Teasing hooks include sensitive topics, repeated corrections, "
                    "personality traits.\n"
                    "If nothing important, respond with NONE.\n"
                    "Return strictly in format: key:value"
                )
            },
            {
                "role": "user",
                "content": user_message
            }
        ]

        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "myra",
                "messages": extraction_prompt,
                "stream": False
            }
        )

        result = response.json()["message"]["content"].strip()

        if result.upper() == "NONE":
            return None

        if ":" in result:
            key, value = result.split(":", 1)
            return key.strip(), value.strip()

        return None

    except:
        return None

# -------------------- CHAT ROUTE --------------------

@app.post("/chat")
def chat(request: ChatRequest):

    global conversation_history, personality_facts

    try:
        user_message = request.message.strip()

        # -------- Deterministic Memory Storage --------
        import re

        match = re.search(r"my name is (.+)", user_message.lower())
        if match:
            name = match.group(1).strip().title()
            cursor.execute(
                "INSERT OR REPLACE INTO personality_memory (key, value) VALUES (?, ?)",
                ("user_name", name)
            )
            conn.commit()
            personality_facts["user_name"] = name

        match = re.search(r"my favorite anime is (.+)", user_message.lower())
        if match:
            anime = match.group(1).strip().title()
            cursor.execute(
                "INSERT OR REPLACE INTO personality_memory (key, value) VALUES (?, ?)",
                ("favorite_anime", anime)
            )
            conn.commit()
            personality_facts["favorite_anime"] = anime

        match = re.search(r"my favorite game is (.+)", user_message.lower())
        if match:
            game = match.group(1).strip().title()
            cursor.execute(
                "INSERT OR REPLACE INTO personality_memory (key, value) VALUES (?, ?)",
                ("favorite_game", game)
            )
            conn.commit()
            personality_facts["favorite_game"] = game

        # ---- Save user message ----
        conversation_history.append({
            "role": "user",
            "content": user_message
        })

        cursor.execute(
            "INSERT INTO conversations (role, content) VALUES (?, ?)",
            ("user", user_message)
        )
        conn.commit()

       

        # ---- Prepare Context ----

        memory_text = "\n".join(
            [f"{k}: {v}" for k, v in personality_facts.items() if k != "relationship_level"]
        )

        # Only inject memory when user asks something related to identity or memory
        memory_keywords = [
            "remember",
            "my name",
            "favorite",
            "anime",
            "game",
            "who am i",
            "what do you know",
            "do you know",
        ]

        inject_memory = any(keyword in user_message.lower() for keyword in memory_keywords)

        if inject_memory and memory_text.strip():
            enriched_user_message = (
                f"User known facts:\n"
                f"{memory_text}\n\n"
                f"User says:\n"
                f"{user_message}"
            )
        else:
            enriched_user_message = user_message

        messages_to_send = [
            {"role": "user", "content": enriched_user_message}
        ]
        # ---- Call Ollama ----
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "myra",
                "messages": messages_to_send,
                "stream": False
            }
        )

        data = response.json()
        reply = data["message"]["content"]

        # ---- Save assistant reply ----
        conversation_history.append({
            "role": "assistant",
            "content": reply
        })

        cursor.execute(
            "INSERT INTO conversations (role, content) VALUES (?, ?)",
            ("assistant", reply)
        )
        conn.commit()

        # ---- Limit memory size ----
        if len(conversation_history) > 40:
            conversation_history[:] = conversation_history[-40:]

        # ---- Generate TTS ----
        tts_text = clean_for_tts(reply)

        with contextlib.redirect_stdout(io.StringIO()):
            audio = model_en.apply_tts(
                text=tts_text,
                speaker='en_0',
                sample_rate=48000
            )

        filename = "output.wav"
        sf.write(filename, audio, 48000)

        return {
            "response": reply,
            "audio_file": filename
        }

    except Exception as e:
        print("Backend error:", e)
        return {
            "response": "Hmm. Something broke. Fix it.",
            "audio_file": None
        }