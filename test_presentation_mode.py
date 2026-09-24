import os
import sys
import time
import unittest
import numpy as np

# Ensure root directory is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi.testclient import TestClient
from main import app, PresenceTracker, get_db
from presentation_manager import presentation_manager, AVATAR_SLIDE_ANCHOR, TRANSITIONS

client = TestClient(app)

class TestPresentationMode(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("\n=======================================================")
        print("  STARTING TEST SUITE: LIVE COLLEGE PRESENTATION MODE  ")
        print("=======================================================")

    def test_01_presence_tracker_separation_anxiety_and_return(self):
        """Tests Mallu absence > 60s triggering MALLU_MISSING and subsequent MALLU_RETURN."""
        print("\n--> [TEST 1] Testing PresenceTracker Separation Anxiety & Return...")
        tracker = PresenceTracker()
        
        # Step 1: Mallu present with class
        detected_faces = [{'name': 'Mallu', 'similarity': 0.9, 'embedding': np.zeros(512), 'bbox': [0,0,10,10]}]
        events = tracker.update(detected_faces, person_count=5)
        self.assertTrue(tracker.mallu_present)
        self.assertEqual(len(events), 0, "No alert when Mallu is present")

        # Step 2: Mallu leaves, but audience remains (person_count >= 1)
        unknown_faces = [{'name': 'Unknown', 'similarity': 0.3, 'embedding': np.ones(512), 'bbox': [0,0,10,10]}]
        tracker.update(unknown_faces, person_count=4)
        self.assertFalse(tracker.mallu_present)

        # Simulate 65 seconds elapsed
        tracker.mallu_last_seen_time = time.time() - 65.0
        events = tracker.update(unknown_faces, person_count=4)
        missing_events = [e for e, _ in events if e == 'MALLU_MISSING']
        self.assertEqual(len(missing_events), 1, "MALLU_MISSING should trigger after 60s absence with audience present")
        self.assertTrue(tracker.mallu_missing_alerted)
        print("    ✓ MALLU_MISSING event triggered correctly after 60s separation.")

        # Step 3: Mallu returns
        events = tracker.update(detected_faces, person_count=5)
        return_events = [e for e, _ in events if e == 'MALLU_RETURN']
        self.assertEqual(len(return_events), 1, "MALLU_RETURN should trigger when Mallu reappears")
        self.assertFalse(tracker.mallu_missing_alerted)
        self.assertTrue(tracker.mallu_present)
        print("    ✓ MALLU_RETURN event triggered correctly upon Mallu's return.")

    def test_02_deck_parsing_and_avatar_slide_detection(self):
        """Verifies slide parsing and that Slide 3 is flagged as the avatar showcase slide."""
        print("\n--> [TEST 2] Testing PPT Deck Parsing & Slide 3 Avatar Flag...")
        loaded = presentation_manager.load_presentation()
        self.assertTrue(loaded, "Default college presentation deck should be loaded")
        self.assertGreaterEqual(len(presentation_manager.slides), 5, "Deck should have at least 5 slides")
        
        slide3 = presentation_manager.slides[2]
        self.assertEqual(slide3["index"], 3)
        self.assertTrue(slide3["has_avatar_image"], "Slide 3 (VRM 3D Avatar) must have has_avatar_image=True")

        # Ensure Slide 1, 2, 4, 5 do not falsely trigger avatar flag
        self.assertFalse(presentation_manager.slides[0]["has_avatar_image"])
        self.assertFalse(presentation_manager.slides[1]["has_avatar_image"])
        self.assertFalse(presentation_manager.slides[3]["has_avatar_image"])
        self.assertFalse(presentation_manager.slides[4]["has_avatar_image"])
        print(f"    ✓ Slide 3 correctly flagged as Avatar Slide: '{slide3['title']}'")

    def test_03_avatar_slide_exact_anchor_delivery(self):
        """Verifies that Slide 3 triggers the exact hardcoded showcase line."""
        print("\n--> [TEST 3] Testing Exact Avatar Slide Line Anchor Rule...")
        presentation_manager.load_presentation()
        presentation_manager.is_presenting = True
        presentation_manager.current_slide_index = 2  # Slide 3

        # Capture output from present_current_slide
        response, _ = presentation_manager.present_current_slide(query_llm_fn=None)
        self.assertEqual(
            response,
            AVATAR_SLIDE_ANCHOR,
            "Avatar slide MUST deliver exact hardcoded line: 'Wait, look at this slide! See how beautiful I look?...'"
        )
        print("    ✓ Avatar Slide delivered exact required showcase line:")
        print(f"      \"{response}\"")

    def test_04_classroom_mass_introduction_trigger(self):
        """Tests '/chat' response to 'Myra, meet my class' with dynamic waifu tease."""
        print("\n--> [TEST 4] Testing Classroom Mass Introduction Trigger via /chat...")
        res = client.post("/chat", json={"message": "Myra, meet my class!"})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("response", data)
        self.assertIsNotNone(data["response"])
        self.assertTrue(len(data["response"]) > 10, "Response should be substantive")
        print(f"    ✓ Classroom Intro Response: \"{data['response']}\"")

    def test_05_faculty_politeness_and_title_detection(self):
        """Tests parsing of honorifics (Professor/Dr) and dynamic polite greeting."""
        print("\n--> [TEST 5] Testing Faculty Politeness & Title Detection...")
        res = client.post("/chat", json={"message": "Myra, please meet Professor Sharma from the CS department."})
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertIn("response", data)
        self.assertTrue(len(data["response"]) > 10)

        # Verify entry in known_people table
        with get_db() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT relationship FROM known_people WHERE name LIKE '%Sharma%'")
            row = cursor.fetchone()
            self.assertIsNotNone(row, "Professor Sharma should be recorded in known_people")
            self.assertIn("Professor", row["relationship"])
        print(f"    ✓ Faculty Greeting Response: \"{data['response']}\"")
        print("    ✓ Professor Sharma recorded with faculty relationship in SQLite.")

    def test_06_autonomous_presentation_lifecycle(self):
        """Tests full lifecycle of /present_ppt endpoint (start -> next -> prev -> stop -> status)."""
        print("\n--> [TEST 6] Testing Autonomous /present_ppt Lifecycle...")
        
        # 1. Start presentation
        start_res = client.post("/present_ppt", json={"action": "start"})
        self.assertEqual(start_res.status_code, 200)
        start_data = start_res.json()
        self.assertTrue(start_data["status"]["is_presenting"])
        self.assertEqual(start_data["status"]["current_slide"], 1)
        print(f"    ✓ Started Slide 1: \"{start_data['response'][:60]}...\"")

        # 2. Advance to Slide 2
        s2_res = client.post("/present_ppt", json={"action": "next"})
        s2_data = s2_res.json()
        self.assertEqual(s2_data["status"]["current_slide"], 2)
        print(f"    ✓ Advanced to Slide 2: \"{s2_data['response'][:60]}...\"")

        # 3. Advance to Slide 3 (Avatar Slide - Exact showcase anchor!)
        s3_res = client.post("/present_ppt", json={"action": "next"})
        s3_data = s3_res.json()
        self.assertEqual(s3_data["status"]["current_slide"], 3)
        self.assertIn("Wait, look at this slide! See how beautiful I look?", s3_data["response"])
        print(f"    ✓ Slide 3 Avatar Showcase verified: \"{s3_data['response'][:60]}...\"")

        # 4. Advance to Slide 4
        s4_res = client.post("/present_ppt", json={"action": "next"})
        s4_data = s4_res.json()
        self.assertEqual(s4_data["status"]["current_slide"], 4)
        print(f"    ✓ Advanced to Slide 4: \"{s4_data['response'][:60]}...\"")

        # 5. Advance to Slide 5
        s5_res = client.post("/present_ppt", json={"action": "next"})
        s5_data = s5_res.json()
        self.assertEqual(s5_data["status"]["current_slide"], 5)
        print(f"    ✓ Advanced to Slide 5: \"{s5_data['response'][:60]}...\"")

        # 6. Stop presentation
        stop_res = client.post("/present_ppt", json={"action": "stop"})
        stop_data = stop_res.json()
        self.assertFalse(stop_data["status"]["is_presenting"])
        print(f"    ✓ Stopped Presentation cleanly: \"{stop_data['response']}\"")

    def test_07_presentation_voice_commands_via_chat(self):
        """Tests natural voice commands for presentation via /chat endpoint."""
        print("\n--> [TEST 7] Testing Voice Navigation Commands via /chat...")

        # Start via chat command
        res = client.post("/chat", json={"message": "Myra, start presentation!"})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json().get("is_presenting", False))
        print("    ✓ 'start presentation' voice command initiated deck.")

        # Next slide via chat command
        res = client.post("/chat", json={"message": "next slide"})
        self.assertEqual(res.status_code, 200)
        status = client.get("/status").json()
        self.assertEqual(status["current_slide"], 2)
        print("    ✓ 'next slide' voice command advanced to Slide 2.")

        # Stop presentation via chat command
        res = client.post("/chat", json={"message": "stop presentation"})
        self.assertEqual(res.status_code, 200)
        status = client.get("/status").json()
        self.assertFalse(status["is_presenting"])
        print("    ✓ 'stop presentation' voice command halted deck.")

if __name__ == "__main__":
    unittest.main()
