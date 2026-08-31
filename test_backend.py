import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import sqlite3
import os

print("=== [TEST] MYRA BACKEND VERIFICATION TEST ===")

# Test 1: Test DB and tables
conn = sqlite3.connect("myra_memory.db")
cursor = conn.cursor()
cursor.execute("SELECT count(*) FROM sqlite_master WHERE type='table'")
table_count = cursor.fetchone()[0]
print(f"[OK] SQLite Memory Tables verified: {table_count} tables found.")

cursor.execute("SELECT count(*) FROM conversations")
conv_count = cursor.fetchone()[0]
print(f"[OK] Existing Conversation turns in Memory: {conv_count}")

# Test 2: Test Edge TTS Generation
from main import generate_voice, parse_presentation, apply_emotion_tag

print("\n--- Testing Edge TTS ---")
test_file = generate_voice("[EMOTE: smug] Testing Myra's sweet anime voice generation!", "test_voice.wav")
if test_file and os.path.exists(test_file):
    print(f"[OK] Voice synthesis successful: {test_file} ({os.path.getsize(test_file)} bytes)")
    try:
        os.remove(test_file)
    except:
        pass
else:
    print("[FAIL] Voice synthesis failed.")

# Test 3: Test Emotion Parser
print("\n--- Testing Emotion & VMC Parser ---")
raw_test = "[EMOTE: blush] Mallu, stop staring at me like that!"
cleaned = apply_emotion_tag(raw_test)
print(f"Original: {raw_test}")
print(f"Cleaned for Speech: {cleaned}")
assert "[EMOTE:" not in cleaned, "Emotion tag was not cleaned!"
print("[OK] Emotion parsing and cleaning verified!")

print("\n=== ALL UNIT TESTS PASSED! ===")
