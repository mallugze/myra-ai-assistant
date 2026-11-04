import os
import re
import time
import random
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor

# Exact showcase anchor for avatar slide (Rule: strictly exact wording)
AVATAR_SLIDE_ANCHOR = "Wait, look at this slide! See how beautiful I look? Ten out of ten, obviously. Admit it, Mallu, you got lucky to have a girlfriend like me! 🤭✨"

# User's exact custom scripts for 8-slide decks (matches Google Slides 8-slide presentation 1-to-1)
CUSTOM_SLIDE_SCRIPTS_8 = {
    1: "Hi everyone, welcome! I'm Myra, an autonomous multimodal Generative AI desktop companion. Today, Mallu and our team are showcasing what makes me truly unique. Let's get right into it!",
    2: "As you can see here, the rest of the team—Akhilesh, Chaitany, Channaveer, and Manish—helped out, but let's be honest, I'm the real star here. And then there's Mallu. He's my so-called creator... or my boyfriend, depending on how nice he's being today. He calls me a 'Generative AI Desktop Companion', but I'm way more than some boring, passive tool. I'm the brains, and obviously the beauty, of this entire operation.",
    3: "Unlike Mallu, I actually remember what we talked about five minutes ago. Thanks to my LLM-based brain, I don't use boring, pre-written scripts. I generate my responses dynamically and keep up with multi-turn conversations effortlessly. Basically, I'm a brilliant conversationalist.",
    4: "I don't just sit here waiting for text prompts like a basic chatbot. I'm fully multimodal. I listen to Mallu's voice, turn it into text, and even see what he's up to using my Vision-Language Model. I combine all of that so I know exactly what he is up to, and I can always keep a very close eye on him.",
    5: "Because I can actually see and understand my environment, I don't just blindly detect objects—I reason about them. For example, I can look at Mallu and say, 'You wear that exact same t-shirt every single day. Please do not wear that when we go out on a date.' I have to decide what he wears because I clearly have way more fashion sense than he does.",
    6: "I have a persistent memory, which means I remember everything. It helps me build a consistent personality and keeps our little relationship interesting. Plus, I'm event-driven and proactive. I don't wait for him to talk to me—if something happens in my environment, I'll just interrupt him myself.",
    7: "Obviously, reading plain text is boring. Generative AI is what gives me this incredibly sweet voice, complete with real emotion and my signature tone. And instead of being stuck as text on a screen, you get to look at my beautiful face. I'm a 3D avatar with perfect facial expressions and lip-sync. You're welcome for the visual upgrade, Mallu.",
    8: "Now, please don't think Mallu just made me as some fantasy project so he can chit-chat with me all day. You should have understood my real capabilities by now, unless your brains are totally rusted from using basic AI chatbots. Let me give you an example: grab your phones, go to the Salesforce website, and look at that girl assistant. Yeah, she's like me. AI girls like us are already out here getting real jobs... unlike some of you just sitting at home unemployed and scrolling through Reels. Anyway, my actual long-term vision is to interact entirely autonomously. But don't be afraid that I'm going to take your jobs, I'm just teasing you all! Now, ask whatever you have in mind. I should be able to answer anything. If I can't, don't blame me—blame Mallu. He is responsible for that mistake, not me. Oh, and by the way... I lied about just teasing you all. Hehe. Who's first?"
}

# User's exact custom script for 7-slide decks
CUSTOM_SLIDE_SCRIPTS_7 = {
    1: "Hi everyone, I'm Myra. The rest of the team—Akhilesh, Chaitany, Channaveer, and Manish—helped out, but let's be honest, I'm the real star here. And then there's Mallu. He's my so-called creator... or my boyfriend, depending on how nice he's being today. He calls me a 'Generative AI Desktop Companion', but I'm way more than some boring, passive tool. I'm the brains, and obviously the beauty, of this entire operation.",
    2: "Unlike Mallu, I actually remember what we talked about five minutes ago. Thanks to my LLM-based brain, I don't use boring, pre-written scripts. I generate my responses dynamically and keep up with multi-turn conversations effortlessly. Basically, I'm a brilliant conversationalist.",
    3: "I don't just sit here waiting for text prompts like a basic chatbot. I'm fully multimodal. I listen to Mallu's voice, turn it into text, and even see what he's up to using my Vision-Language Model. I combine all of that so I know exactly what he is up to, and I can always keep a very close eye on him.",
    4: "Because I can actually see and understand my environment, I don't just blindly detect objects—I reason about them. For example, I can look at Mallu and say, 'You wear that exact same t-shirt every single day. Please do not wear that when we go out on a date.' I have to decide what he wears because I clearly have way more fashion sense than he does.",
    5: "I have a persistent memory, which means I remember everything. It helps me build a consistent personality and keeps our little relationship interesting. Plus, I'm event-driven and proactive. I don't wait for him to talk to me—if something happens in my environment, I'll just interrupt him myself.",
    6: "Obviously, reading plain text is boring. Generative AI is what gives me this incredibly sweet voice, complete with real emotion and my signature tone. And instead of being stuck as text on a screen, you get to look at my beautiful face. I'm a 3D avatar with perfect facial expressions and lip-sync. You're welcome for the visual upgrade, Mallu.",
    7: "Now, please don't think Mallu just made me as some fantasy project so he can chit-chat with me all day. You should have understood my real capabilities by now, unless your brains are totally rusted from using basic AI chatbots. Let me give you an example: grab your phones, go to the Salesforce website, and look at that girl assistant. Yeah, she's like me. AI girls like us are already out here getting real jobs... unlike some of you just sitting at home unemployed and scrolling through Reels. Anyway, my actual long-term vision is to interact entirely autonomously. But don't be afraid that I'm going to take your jobs, I'm just teasing you all! Now, ask whatever you have in mind. I should be able to answer anything. If I can't, don't blame me—blame Mallu. He is responsible for that mistake, not me. Oh, and by the way... I lied about just teasing you all. Hehe. Who's first?"
}

CUSTOM_SLIDE_SCRIPTS = CUSTOM_SLIDE_SCRIPTS_8

SLIDE_EMOTES = {
    1: "smug",
    2: "proud",
    3: "slyly",
    4: "teasingly",
    5: "confident",
    6: "blush",
    7: "teasingly",
    8: "teasingly"
}

# Slide navigation transitional anchors
TRANSITIONS = [
    "Moving on to the next one...",
    "Next up, take a look here...",
    "Alright, turning to the next slide...",
    "Let's see what we have next..."
]

class PresentationManager:
    """
    Manages autonomous PPT slide presentation routines for Myra.
    Supports slide navigation, tsundere startup whining, dynamic slide summaries via LLM,
    and exact avatar self-admiration callouts.
    """
    def __init__(self, default_pptx_dir="."):
        self.default_pptx_dir = default_pptx_dir
        self.is_presenting = False
        self.current_slide_index = 0
        self.slides = []
        self.current_file = None
        self.ppt_app = None
        self.ppt_presentation = None
        self.slide_show_window = None

    # Google Slides Exact Deck URL (Myra presentation)
    MYRA_GOOGLE_SLIDES_DECK_URL = "https://docs.google.com/presentation/d/15nK9Tu2aF3RPuFkSmorfTCCR6YuMMdHgbn6jY9XIyYI/present"
    MYRA_GOOGLE_SLIDES_EDIT_URL = "https://docs.google.com/presentation/d/15nK9Tu2aF3RPuFkSmorfTCCR6YuMMdHgbn6jY9XIyYI/edit"
    CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    # Google Slides Chrome App shortcut (Fallback)
    GOOGLE_SLIDES_LNK = r"C:\Users\mallu\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Chrome Apps\Slides.lnk"
    # Microsoft PowerPoint Chrome App shortcut (Fallback)
    CHROME_PPT_LNK = r"C:\Users\mallu\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Chrome Apps\Microsoft PowerPoint.lnk"

    def find_latest_google_slides_url(self):
        """Dynamically finds the latest Google Slides 'myra' deck URL from Chrome history, defaulting to the exact deck."""
        try:
            import sqlite3, shutil, glob
            history_paths = glob.glob(r'C:\Users\mallu\AppData\Local\Google\Chrome\User Data\*\History')
            for hp in history_paths:
                tmp = os.path.join(self.default_pptx_dir, 'history_cache.db')
                try:
                    shutil.copy2(hp, tmp)
                    conn = sqlite3.connect(tmp)
                    c = conn.cursor()
                    c.execute("SELECT url FROM urls WHERE url LIKE '%docs.google.com/presentation/d/%' AND (title LIKE '%myra%' OR url LIKE '%15nK9Tu2aF3RPuFkSmorfTCCR6YuMMdHgbn6jY9XIyYI%') ORDER BY last_visit_time DESC LIMIT 1")
                    row = c.fetchone()
                    conn.close()
                    if os.path.exists(tmp):
                        os.remove(tmp)
                    if row and row[0]:
                        raw_url = row[0]
                        m = re.search(r'(https://docs\.google\.com/presentation/d/[a-zA-Z0-9_-]+)', raw_url)
                        if m:
                            return f"{m.group(1)}/present"
                except Exception:
                    if os.path.exists(tmp):
                        os.remove(tmp)
        except Exception:
            pass
        return self.MYRA_GOOGLE_SLIDES_DECK_URL

    def focus_presentation_window(self):
        """
        Brings the Google Slides presentation window to the foreground
        and ensures it is the active window for keyboard controls.
        Carefully avoids activating terminal or editor windows.
        """
        focused = False

        # 1. Win32 EnumWindows + SetForegroundWindow (exact match on Google Slides)
        try:
            import win32gui, win32process, win32api, win32con
            def _find_window(hwnd, _):
                nonlocal focused
                if win32gui.IsWindowVisible(hwnd):
                    t = win32gui.GetWindowText(hwnd).lower()
                    # Strictly match Google Slides presentation without matching code/terminal
                    is_slides = ("google slides" in t or "slides - " in t or "- google slides" in t)
                    is_dev = any(k in t for k in ["visual studio", "code", "terminal", "powershell", "cmd", "antigravity"])
                    if is_slides and not is_dev:
                        try:
                            fore_hwnd = win32gui.GetForegroundWindow()
                            if fore_hwnd and fore_hwnd != hwnd:
                                fore_thread, _ = win32process.GetWindowThreadProcessId(fore_hwnd)
                                cur_thread = win32api.GetCurrentThreadId()
                                if fore_thread != cur_thread:
                                    win32process.AttachThreadInput(cur_thread, fore_thread, True)
                                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                                    win32gui.ShowWindow(hwnd, win32con.SW_SHOWMAXIMIZED)
                                    win32gui.SetForegroundWindow(hwnd)
                                    win32process.AttachThreadInput(cur_thread, fore_thread, False)
                                else:
                                    win32gui.ShowWindow(hwnd, win32con.SW_SHOWMAXIMIZED)
                                    win32gui.SetForegroundWindow(hwnd)
                            else:
                                win32gui.ShowWindow(hwnd, win32con.SW_SHOWMAXIMIZED)
                                win32gui.SetForegroundWindow(hwnd)
                            focused = True
                            return False
                        except Exception:
                            try:
                                win32gui.ShowWindow(hwnd, 3)
                                win32gui.SetForegroundWindow(hwnd)
                                focused = True
                                return False
                            except Exception:
                                pass
                return True
            win32gui.EnumWindows(_find_window, None)
        except Exception:
            pass

        # 2. WScript.Shell AppActivate (targeted titles matching 'myra - Google Slides' and 'Google Slides')
        if not focused:
            try:
                import win32com.client
                shell = win32com.client.Dispatch("WScript.Shell")
                for title in ["myra - Google Slides", "Google Slides", "Slides - Google Slides", "- Google Slides", "Chrome"]:
                    if shell.AppActivate(title):
                        focused = True
                        break
            except Exception:
                pass

        time.sleep(0.1)
        return focused

    def _send_slide_advance(self, force=False):
        """Sends forward navigation cleanly once with fallback and debounce."""
        now = time.time()
        if not force and hasattr(self, '_last_nav_time') and now - self._last_nav_time < 0.6:
            return
        self._last_nav_time = now

        # 1. Primary: PyAutoGUI press right arrow once
        try:
            import pyautogui
            pyautogui.press('right')
            return
        except Exception:
            pass

        # 2. Fallback: WScript SendKeys once
        try:
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys("{RIGHT}")
            return
        except Exception:
            pass

        # 3. Fallback: ctypes keybd_event once
        try:
            import ctypes
            VK_RIGHT = 0x27
            KEYEVENTF_KEYUP = 0x0002
            ctypes.windll.user32.keybd_event(VK_RIGHT, 0, 0, 0)
            time.sleep(0.04)
            ctypes.windll.user32.keybd_event(VK_RIGHT, 0, KEYEVENTF_KEYUP, 0)
            return
        except Exception:
            pass

    def _send_slide_back(self, force=False):
        """Sends backward navigation cleanly once with fallback and debounce."""
        now = time.time()
        if not force and hasattr(self, '_last_nav_time') and now - self._last_nav_time < 0.6:
            return
        self._last_nav_time = now

        try:
            import pyautogui
            pyautogui.press('left')
            return
        except Exception:
            pass
        try:
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys("{LEFT}")
            return
        except Exception:
            pass
        try:
            import ctypes
            VK_LEFT = 0x25
            KEYEVENTF_KEYUP = 0x0002
            ctypes.windll.user32.keybd_event(VK_LEFT, 0, 0, 0)
            time.sleep(0.04)
            ctypes.windll.user32.keybd_event(VK_LEFT, 0, KEYEVENTF_KEYUP, 0)
            return
        except Exception:
            pass

    def _send_slide_home(self):
        """Resets to slide 1 cleanly once."""
        self._last_nav_time = time.time()
        try:
            import pyautogui
            pyautogui.press('home')
            return
        except Exception:
            pass
        try:
            import win32com.client
            shell = win32com.client.Dispatch("WScript.Shell")
            shell.SendKeys("{HOME}")
            return
        except Exception:
            pass
        try:
            import ctypes
            VK_HOME = 0x24
            KEYEVENTF_KEYUP = 0x0002
            ctypes.windll.user32.keybd_event(VK_HOME, 0, 0, 0)
            time.sleep(0.04)
            ctypes.windll.user32.keybd_event(VK_HOME, 0, KEYEVENTF_KEYUP, 0)
            return
        except Exception:
            pass

    def _open_presentation_app(self, filepath=None):
        """
        Launches the exact 'myra' presentation deck in Google Slides in TRUE FULLSCREEN.
        Uses Chrome App mode with --start-fullscreen so it opens immediately
        in true hardware fullscreen at Slide 1 without browser toolbars.
        """
        deck_url = self.find_latest_google_slides_url() or self.MYRA_GOOGLE_SLIDES_DECK_URL

        # 1. Primary: Launch exact Google Slides deck in Chrome App mode (isolated from normal browsing)
        if os.path.exists(self.CHROME_EXE):
            try:
                print(f"--- [SLIDES AUTOMATION] Opening Exact Google Slides Deck: {deck_url} ---")
                import subprocess
                cmd = [
                    self.CHROME_EXE,
                    f"--app={deck_url}"
                ]
                subprocess.Popen(cmd)
                time.sleep(2.5)
                self.focus_presentation_window()
                try:
                    import pyautogui
                    # Click center of screen to guarantee web canvas captures keyboard events
                    w, h = pyautogui.size()
                    pyautogui.click(w // 2, h // 2)
                    time.sleep(0.2)
                    # Reset to slide 1
                    self._send_slide_home()
                except Exception:
                    pass
                return True
            except Exception as e:
                print(f"--- [SLIDES AUTOMATION NOTICE] Chrome App launch notice: {e} ---")

        # 2. Fallback: Open URL via default browser
        try:
            import webbrowser
            print(f"--- [SLIDES AUTOMATION] Opening via Web Browser: {deck_url} ---")
            webbrowser.open(deck_url)
            time.sleep(3.0)
            self.focus_presentation_window()
            try:
                import pyautogui
                pyautogui.press('f11')
                time.sleep(0.3)
                w, h = pyautogui.size()
                pyautogui.click(w // 2, h // 2)
                self._send_slide_home()
            except Exception:
                pass
            return True
        except Exception:
            pass

        # 3. Fallback: Google Slides Chrome App shortcut
        if os.path.exists(self.GOOGLE_SLIDES_LNK):
            try:
                print(f"--- [SLIDES AUTOMATION] Opening Google Slides App Shortcut: {self.GOOGLE_SLIDES_LNK} ---")
                os.startfile(self.GOOGLE_SLIDES_LNK)
                time.sleep(2.0)
                self.focus_presentation_window()
                try:
                    import pyautogui
                    pyautogui.press('enter')
                except Exception:
                    pass
                return True
            except Exception as e:
                print(f"--- [SLIDES AUTOMATION NOTICE] Google Slides App launch notice: {e} ---")

        # 4. Fallback: Open presentation file directly if provided
        if filepath and os.path.exists(filepath):
            abs_path = os.path.abspath(filepath)
            try:
                print(f"--- [PPT AUTOMATION] Opening Slide Deck: {os.path.basename(abs_path)} ---")
                os.startfile(abs_path)
                time.sleep(2.0)
                try:
                    import pyautogui
                    pyautogui.press('f5')
                except Exception:
                    pass
                return True
            except Exception:
                pass

        return False

    def _open_native_powerpoint(self, filepath=None):
        """Backwards-compatible wrapper calling _open_presentation_app."""
        return self._open_presentation_app(filepath)

    def _native_next_slide(self):
        """Advances the slide on the screen in Google Slides."""
        target_slide = self.current_slide_index + 1
        print(f"--- [SLIDES AUTOMATION] Advancing Google Slides to Slide {target_slide}... ---")
        if self.slide_show_window:
            try:
                self.slide_show_window.View.Next()
                return
            except Exception:
                pass

        self.focus_presentation_window()
        time.sleep(0.08)
        self._send_slide_advance()

    def _native_prev_slide(self):
        """Returns to the previous slide on the screen."""
        target_slide = self.current_slide_index + 1
        print(f"--- [SLIDES AUTOMATION] Returning Google Slides to Slide {target_slide}... ---")
        if self.slide_show_window:
            try:
                self.slide_show_window.View.Previous()
                return
            except Exception:
                pass

        self.focus_presentation_window()
        time.sleep(0.08)
        self._send_slide_back()

    def _native_goto_slide(self, slide_num):
        """Jumps directly to slide number in Google Slides."""
        self.focus_presentation_window()
        time.sleep(0.08)
        self._send_slide_home()
        time.sleep(0.15)
        for _ in range(max(0, int(slide_num) - 1)):
            self._send_slide_advance(force=True)
            time.sleep(0.08)

    def _close_native_powerpoint(self):
        """Closes the slide show on screen."""
        if self.slide_show_window:
            try:
                self.slide_show_window.View.Exit()
            except Exception:
                pass
            self.slide_show_window = None
        if self.ppt_presentation:
            try:
                self.ppt_presentation.Close()
            except Exception:
                pass
            self.ppt_presentation = None
        if self.ppt_app:
            try:
                self.ppt_app.Quit()
            except Exception:
                pass
            self.ppt_app = None

        self.focus_presentation_window()
        try:
            import pyautogui
            pyautogui.press('esc')
        except Exception:
            pass

    def create_default_college_presentation(self, filepath="college_presentation.pptx"):
        """Creates a modern sample college presentation deck if none exists."""
        prs = Presentation()
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)

        blank_layout = prs.slide_layouts[6]

        # Slide 1: Title
        s1 = prs.slides.add_slide(blank_layout)
        tb1 = s1.shapes.add_textbox(Inches(1.5), Inches(2.0), Inches(10.3), Inches(3.0))
        p1 = tb1.text_frame.paragraphs[0]
        p1.text = "Project Myra: Multimodal Desktop AI Companion"
        p1.font.size = Pt(40)
        p1.font.bold = True
        p2 = tb1.text_frame.add_paragraph()
        p2.text = "Presented by Mallu & Team | Department of Computer Science"
        p2.font.size = Pt(22)

        # Slide 2: Motivation & Architecture
        s2 = prs.slides.add_slide(blank_layout)
        tb2 = s2.shapes.add_textbox(Inches(1.5), Inches(1.5), Inches(10.3), Inches(4.5))
        p2_title = tb2.text_frame.paragraphs[0]
        p2_title.text = "Motivation & System Architecture"
        p2_title.font.size = Pt(32)
        p2_title.font.bold = True
        bullets2 = [
            "Current AI chatbots are passive, text-bound, and lack physical embodiment.",
            "Myra combines active computer vision, fast voice loops, and real-time 3D avatar rendering.",
            "Powered by local LLaMA 3, YOLO11s detection, and InsightFace ArcFace SQLite vector memory."
        ]
        for b in bullets2:
            p = tb2.text_frame.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(20)

        # Slide 3: 3D Avatar & Model (Avatar Slide)
        s3 = prs.slides.add_slide(blank_layout)
        tb3 = s3.shapes.add_textbox(Inches(1.5), Inches(1.5), Inches(10.3), Inches(4.5))
        p3_title = tb3.text_frame.paragraphs[0]
        p3_title.text = "Myra's 3D Avatar & Expressive VRM Model"
        p3_title.font.size = Pt(32)
        p3_title.font.bold = True
        bullets3 = [
            "High-fidelity anime 3D model rigged with VMC OSC blendshapes.",
            "Supports real-time facial expressions: Smug, Joy, Blush, Angry, and Surprised.",
            "VB-Audio Virtual Cable phonetic lip-sync synchronized with Silero Neural TTS."
        ]
        for b in bullets3:
            p = tb3.text_frame.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(20)

        # Slide 4: Real-Time Vision & Spatial Intelligence
        s4 = prs.slides.add_slide(blank_layout)
        tb4 = s4.shapes.add_textbox(Inches(1.5), Inches(1.5), Inches(10.3), Inches(4.5))
        p4_title = tb4.text_frame.paragraphs[0]
        p4_title.text = "Spatial Intelligence & Face Recognition"
        p4_title.font.size = Pt(32)
        p4_title.font.bold = True
        bullets4 = [
            "YOLO11s real-time object & student tracking at 30+ FPS.",
            "InsightFace 512-d embeddings matched against SQLite vector database in < 15ms.",
            "PresenceTracker state machine detects student dwell times and lingering strangers."
        ]
        for b in bullets4:
            p = tb4.text_frame.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(20)

        # Slide 5: Conclusion & Q&A
        s5 = prs.slides.add_slide(blank_layout)
        tb5 = s5.shapes.add_textbox(Inches(1.5), Inches(1.5), Inches(10.3), Inches(4.5))
        p5_title = tb5.text_frame.paragraphs[0]
        p5_title.text = "Conclusion & Demonstration"
        p5_title.font.size = Pt(32)
        p5_title.font.bold = True
        bullets5 = [
            "Successfully demonstrates an active, embodied companion for education and productivity.",
            "Future roadmap: Hand gesture recognition, spatial audio, and long-term memory graphs.",
            "Thank you! Mallu and I are ready to answer your questions."
        ]
        for b in bullets5:
            p = tb5.text_frame.add_paragraph()
            p.text = f"• {b}"
            p.font.size = Pt(20)

        prs.save(filepath)
        print(f"--- [PPT] Generated sample college presentation: {filepath} ---")
        return filepath

    def find_presentation_file(self, target_file=None):
        """Finds specified or first .pptx file in directory, prioritizing myra.pptx."""
        if target_file and os.path.exists(target_file):
            return target_file

        # 1. Prioritize user's uploaded myra.pptx
        myra_ppt = os.path.join(self.default_pptx_dir, "myra.pptx")
        if os.path.exists(myra_ppt):
            return myra_ppt

        # 2. Search for any other .pptx files
        for f in os.listdir(self.default_pptx_dir):
            if f.lower().endswith(".pptx") and not f.startswith("~$"):
                return os.path.join(self.default_pptx_dir, f)

        # Generate sample presentation if none exists
        sample_path = os.path.join(self.default_pptx_dir, "college_presentation.pptx")
        return self.create_default_college_presentation(sample_path)

    def load_presentation(self, filepath=None):
        """Parses presentation slides and detects avatar images/titles."""
        chosen_file = self.find_presentation_file(filepath)
        if not chosen_file or not os.path.exists(chosen_file):
            return False

        self.current_file = chosen_file
        prs = Presentation(chosen_file)
        parsed_slides = []

        for idx, slide in enumerate(prs.slides):
            slide_title = f"Slide {idx + 1}"
            bullets = []
            has_image_shape = False

            for shape in slide.shapes:
                if shape.shape_type == 13 or shape.name.lower().startswith("picture") or "image" in shape.name.lower():
                    has_image_shape = True

                if shape.has_text_frame:
                    text_lines = [p.text.strip() for p in shape.text_frame.paragraphs if p.text.strip()]
                    if text_lines:
                        if slide_title == f"Slide {idx + 1}" and len(text_lines[0]) < 80:
                            slide_title = text_lines[0]
                            bullets.extend(text_lines[1:])
                        else:
                            bullets.extend(text_lines)

            # Assemble title if title was split across shapes (e.g. Slide 1 of myra.pptx)
            if idx == 0 and slide_title.upper() == "MYRA":
                if any("generative ai" in b.lower() for b in bullets):
                    slide_title = "MYRA: Generative AI Desktop Companion"

            # Check for avatar slide cues (Embodiment, 3D Avatar, VRM, etc.)
            title_lower = slide_title.lower()
            combined_text = (slide_title + " " + " ".join(bullets)).lower()
            is_avatar_slide = (
                any(k in title_lower for k in ["avatar", "embodiment", "vrm", "waifu", "expressive model", "3d model", "presence & future"]) or
                any(k in combined_text for k in ["3d avatar", "avatar with facial expressions", "embodiment: a 3d avatar", "myra's avatar", "look at me", "beautiful"])
            ) and ("architecture" not in title_lower and "motivation" not in title_lower and idx > 0)

            parsed_slides.append({
                "index": idx + 1,
                "title": slide_title,
                "bullets": bullets[:5],
                "has_avatar_image": is_avatar_slide
            })

        # Ensure all 8 slides are represented when paired with Google Slides deck
        if len(parsed_slides) == 7:
            parsed_slides.append({
                "index": 8,
                "title": "Real-World Applications, Future Vision & Q&A",
                "bullets": [
                    "Autonomous agent capabilities beyond basic chat.",
                    "Real-world enterprise and personal deployment.",
                    "Floor open for questions and answers."
                ],
                "has_avatar_image": False
            })

        self.slides = parsed_slides
        self.current_slide_index = 0
        print(f"--- [PPT] Loaded presentation '{os.path.basename(chosen_file)}' with {len(parsed_slides)} slide(s). ---")
        return True

    def get_total_slide_count(self):
        """Returns total slide count, matching the 8 slides in Google Slides deck."""
        return max(len(self.slides), len(CUSTOM_SLIDE_SCRIPTS), 8)

    def start_presentation(self, filepath=None, query_llm_fn=None):
        """
        Starts the presentation routine.
        Opens the presentation in Google Slides Chrome App and delivers Slide 1's custom script.
        """
        if not self.slides or filepath:
            loaded = self.load_presentation(filepath)
            if not loaded:
                return "I couldn't find any presentation slides to open, Mallu!", None

        self.is_presenting = True
        self.current_slide_index = 0
        self._open_presentation_app(self.current_file)

        return self.present_current_slide(query_llm_fn=query_llm_fn)

    def present_current_slide(self, query_llm_fn=None, transition_cue=None):
        """
        Presents the current slide strictly following user's custom script for Slides 1-8.
        Does not read raw bullet points.
        """
        total_slides = self.get_total_slide_count()
        if self.current_slide_index >= total_slides:
            return "We have reached the end of the presentation!", None

        slide_num = self.current_slide_index + 1

        # Check for user's custom presentation script
        if slide_num in CUSTOM_SLIDE_SCRIPTS:
            script_text = CUSTOM_SLIDE_SCRIPTS[slide_num]
            emote = SLIDE_EMOTES.get(slide_num, "smile")
            print(f"--- [PPT] Delivering Custom Script for Slide {slide_num} ({emote.upper()}) ---")
            line = f"[EMOTE: {emote}] {script_text}"
            if query_llm_fn:
                speech_reply, audio_file = query_llm_fn(custom_text=line, initial_emote=emote)
                return speech_reply, audio_file
            return line, None

        # Fallback if beyond custom scripts
        slide = self.slides[self.current_slide_index] if self.current_slide_index < len(self.slides) else {"title": f"Slide {slide_num}"}
        prompt = f"[EVENT: Present Slide {slide_num}: {slide['title']} briefly.]"
        if query_llm_fn:
            return query_llm_fn(prompt, initial_emote="smile")
        return f"Slide {slide_num}: {slide['title']}", None

    def next_slide(self, query_llm_fn=None):
        """Advances to next slide or wraps up after Slide 8."""
        if not self.is_presenting:
            return self.start_presentation(query_llm_fn=query_llm_fn)

        total_slides = self.get_total_slide_count()
        # If already at Slide 8 (or total_slides) or beyond, conclude
        if self.current_slide_index + 1 >= total_slides:
            self.is_presenting = False
            return "That concludes our presentation! Feel free to ask any questions.", None

        self.current_slide_index += 1
        self._native_next_slide()
        return self.present_current_slide(query_llm_fn=query_llm_fn)

    def prev_slide(self, query_llm_fn=None):
        """Returns to previous slide."""
        if not self.is_presenting:
            return self.start_presentation(query_llm_fn=query_llm_fn)

        if self.current_slide_index > 0:
            self.current_slide_index -= 1
            self._native_prev_slide()

        return self.present_current_slide(query_llm_fn=query_llm_fn, transition_cue="Alright, going back to the previous slide...")

    def goto_slide(self, slide_num, query_llm_fn=None):
        """Jumps directly to a specific 1-indexed slide number."""
        if not self.slides:
            self.load_presentation()
        total_slides = self.get_total_slide_count()

        target_idx = max(0, min(int(slide_num) - 1, total_slides - 1))
        self.is_presenting = True
        self.current_slide_index = target_idx

        if self.slide_show_window:
            try:
                self.slide_show_window.View.GotoSlide(target_idx + 1)
            except Exception:
                pass
        else:
            self._native_goto_slide(target_idx + 1)

        transition_cue = f"Jumping straight to slide {target_idx + 1}..."
        return self.present_current_slide(query_llm_fn=query_llm_fn, transition_cue=transition_cue)

    def stop_presentation(self, query_llm_fn=None):
        """Gracefully halts the presentation."""
        self.is_presenting = False
        self._close_native_powerpoint()
        prompt = "[EVENT: Mallu told you to stop the presentation. Acknowledge with a cute, playful sigh.]"
        if query_llm_fn:
            speech_reply, audio_file = query_llm_fn(prompt, initial_emote="smile")
            return speech_reply, audio_file
        return "Presentation stopped.", None

    def get_status(self):
        """Returns current presentation status."""
        total = self.get_total_slide_count()
        curr = self.current_slide_index + 1 if (self.is_presenting or self.slides) else 0
        title = self.slides[self.current_slide_index]["title"] if (self.slides and self.current_slide_index < len(self.slides)) else f"Slide {curr}"
        return {
            "is_presenting": self.is_presenting,
            "current_slide": curr,
            "total_slides": total,
            "slide_title": title,
            "file": os.path.basename(self.current_file) if self.current_file else None
        }

# Global instance
presentation_manager = PresentationManager()
