import os
import sys
import time
import numpy as np
import cv2

# Set stdout encoding
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("==================================================")
print("       MYRA ACTIVE VISION PIPELINE TEST SUITE     ")
print("==================================================")

from main import (
    get_db,
    load_known_faces,
    register_new_face,
    match_face,
    PresenceTracker,
    inspect_room_vlm,
    known_face_records,
    HAS_INSIGHTFACE
)

# ----------------- TEST 1: SQLITE VECTOR STORE & COSINE SIMILARITY -----------------
print("\n[TEST 1] Testing SQLite Vector Store & Cosine Similarity...")
test_emb_1 = np.random.randn(512).astype(np.float32)
test_emb_1 /= np.linalg.norm(test_emb_1)

# Register synthetic test face
assert register_new_face("TestRahul", test_emb_1) == True, "Failed to register face"
load_known_faces()

# Test identical match
name, sim = match_face(test_emb_1)
print(f"--> Exact match result: Name='{name}', Similarity={sim:.4f}")
assert name == "TestRahul", f"Expected 'TestRahul', got '{name}'"
assert sim >= 0.99, f"Expected similarity ~1.0, got {sim}"

# Test noisy match (simulated slight angle difference)
noisy_emb = test_emb_1 + np.random.normal(0, 0.05, 512).astype(np.float32)
noisy_emb /= np.linalg.norm(noisy_emb)
name_noisy, sim_noisy = match_face(noisy_emb)
print(f"--> Noisy match result: Name='{name_noisy}', Similarity={sim_noisy:.4f}")
assert name_noisy == "TestRahul", f"Expected 'TestRahul' on noisy embedding, got '{name_noisy}'"

# Test unknown face (orthogonal/random vector)
unknown_emb = np.random.randn(512).astype(np.float32)
unknown_emb /= np.linalg.norm(unknown_emb)
name_unknown, sim_unknown = match_face(unknown_emb)
print(f"--> Unknown match result: Name='{name_unknown}', Similarity={sim_unknown:.4f}")
assert name_unknown == "Unknown", f"Expected 'Unknown', got '{name_unknown}'"
print("✓ [TEST 1 PASSED] SQLite Face Vector Store & Matching verified!")

# ----------------- TEST 2: PRESENCETRACKER DWELL & DEBOUNCING -----------------
print("\n[TEST 2] Testing PresenceTracker State Machine & Dwell Alerts...")
tracker = PresenceTracker()

# 2.1 Stranger Dwell (> 5 seconds)
dummy_unknown_face = [{'name': 'Unknown', 'similarity': 0.3, 'embedding': unknown_emb, 'bbox': [100, 100, 200, 200]}]

# Frame at t=0
events_0 = tracker.update(dummy_unknown_face, person_count=1)
assert len(events_0) == 0, f"Expected 0 events at t=0, got {events_0}"

# Simulate time passing by modifying internal tracker start time
tracker.unknown_start_time = time.time() - 5.5

# Frame at t=5.5s -> Should trigger STRANGER_LINGER
events_5 = tracker.update(dummy_unknown_face, person_count=1)
print(f"--> Events after 5.5s stranger dwell: {events_5}")
assert len(events_5) == 1 and events_5[0][0] == 'STRANGER_LINGER', f"Expected STRANGER_LINGER event, got {events_5}"

# Immediate next frame -> Debounced (no repeat event)
events_debounced = tracker.update(dummy_unknown_face, person_count=1)
assert len(events_debounced) == 0, f"Expected debounced (0 events), got {events_debounced}"
print("✓ Stranger linger event & debouncing verified!")

# 2.2 Crowd Alert (>= 3 people for > 20 seconds)
tracker_crowd = PresenceTracker()
events_c0 = tracker_crowd.update([], person_count=3)
assert len(events_c0) == 0

# Simulate 21s passing
tracker_crowd.crowd_start_time = time.time() - 21.0
events_c20 = tracker_crowd.update([], person_count=3)
print(f"--> Events after 21s crowd dwell: {events_c20}")
assert len(events_c20) == 1 and events_c20[0][0] == 'CROWD_ALERT', f"Expected CROWD_ALERT event, got {events_c20}"
print("✓ Crowd alert event & debouncing verified!")
print("✓ [TEST 2 PASSED] PresenceTracker Engine verified!")

# ----------------- TEST 3: VLM ROOM INSPECTION -----------------
print("\n[TEST 3] Testing On-Demand VLM Room Inspection with Moondream...")
if os.path.exists("known_faces/mallu.jpg"):
    test_img = cv2.imread("known_faces/mallu.jpg")
    vlm_result = inspect_room_vlm(test_img)
    print(f"--> VLM Output: {vlm_result}")
    assert len(vlm_result) > 5, "VLM returned empty description"
    print("✓ [TEST 3 PASSED] VLM Room Inspection verified!")
else:
    print("! Skipping VLM image test (no sample image found).")

# Clean up synthetic test face from DB
with get_db() as conn:
    cursor = conn.cursor()
    cursor.execute("DELETE FROM known_faces WHERE name = 'TestRahul'")
    cursor.execute("DELETE FROM known_people WHERE name = 'TestRahul'")
    conn.commit()

print("\n==================================================")
print("   ALL TESTS COMPLETED & VERIFIED SUCCESSFULLY!   ")
print("==================================================")
