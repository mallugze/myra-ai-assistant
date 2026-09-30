"""
Audio Device Router for Myra AI Assistant
Intelligently discovers connected headphones, headsets, and microphones.
Prioritizes external/headphone devices over laptop built-in microphones/speakers,
and manages dual-channel playback (Headphones + VB-Cable for VSeeFace lip-sync).
"""
import os
import time
import threading
import sounddevice as sd
import soundfile as sf

HEADPHONE_KEYWORDS = [
    'headset', 'headphone', 'rockerz', 'earphone',
    'bluetooth', 'wireless', 'usb', 'airpod', 'buds', 'mic'
]

VIRTUAL_KEYWORDS = [
    'cable', 'virtual', 'mapper', 'primary sound', 'stereo mix'
]

def is_virtual_device(name: str) -> bool:
    return any(v in name.lower() for v in VIRTUAL_KEYWORDS)

def is_external_or_headphone(name: str) -> bool:
    name_lower = name.lower()
    return any(k in name_lower for k in HEADPHONE_KEYWORDS) and not is_virtual_device(name)

def detect_audio_devices(verbose: bool = True) -> dict:
    """
    Scans all system audio devices.
    Returns:
        {
            'input_device': int,
            'input_name': str,
            'is_headphone_mic': bool,
            'output_device': int,
            'output_name': str,
            'is_headphones': bool,
            'vb_cable_device': int or None
        }
    """
    try:
        devs = sd.query_devices()
        default_in = sd.default.device[0]
        default_out = sd.default.device[1]
    except Exception as e:
        if verbose:
            print(f"--- [AUDIO ROUTER] Notice querying devices: {e} ---")
        return {
            'input_device': None,
            'input_name': 'System Default',
            'is_headphone_mic': False,
            'output_device': None,
            'output_name': 'System Default',
            'is_headphones': False,
            'vb_cable_device': None
        }

    best_out_idx = None
    best_out_name = None
    is_headphones = False

    best_in_idx = None
    best_in_name = None
    is_headphone_mic = False

    vb_cable_idx = None

    # 1. Detect VB-Cable for avatar lip-sync
    for i, d in enumerate(devs):
        if d['max_output_channels'] > 0:
            if 'cable input' in d['name'].lower() and vb_cable_idx is None:
                vb_cable_idx = i
                break

    # 2. Find Output: Priority 1 = Headphones / Headset
    for i, d in enumerate(devs):
        if d['max_output_channels'] > 0:
            if is_external_or_headphone(d['name']) and best_out_idx is None:
                best_out_idx = i
                best_out_name = d['name']
                is_headphones = True
                break

    # Output: Priority 2 = Realtek / physical speakers
    if best_out_idx is None:
        for i, d in enumerate(devs):
            if d['max_output_channels'] > 0 and not is_virtual_device(d['name']):
                best_out_idx = i
                best_out_name = d['name']
                is_headphones = False
                break

    # Output: Priority 3 = System Default Output
    if best_out_idx is None:
        best_out_idx = default_out
        best_out_name = devs[default_out]['name'] if default_out is not None and default_out >= 0 else 'Default Output'
        is_headphones = is_external_or_headphone(best_out_name)

    # 3. Find Input: Priority 1 = Headset / External Mic
    for i, d in enumerate(devs):
        if d['max_input_channels'] > 0:
            if is_external_or_headphone(d['name']) and best_in_idx is None:
                best_in_idx = i
                best_in_name = d['name']
                is_headphone_mic = True
                break

    # Input: Priority 2 = Physical Built-in Microphone Array
    if best_in_idx is None:
        for i, d in enumerate(devs):
            if d['max_input_channels'] > 0 and not is_virtual_device(d['name']):
                best_in_idx = i
                best_in_name = d['name']
                is_headphone_mic = False
                break

    # Input: Priority 3 = System Default Input
    if best_in_idx is None:
        best_in_idx = default_in
        best_in_name = devs[default_in]['name'] if default_in is not None and default_in >= 0 else 'Default Input'
        is_headphone_mic = is_external_or_headphone(best_in_name)

    config = {
        'input_device': best_in_idx,
        'input_name': best_in_name,
        'is_headphone_mic': is_headphone_mic,
        'output_device': best_out_idx,
        'output_name': best_out_name,
        'is_headphones': is_headphones,
        'vb_cable_device': vb_cable_idx
    }

    if verbose:
        print("\n+==================================================================+")
        print("|           [AUDIO] MYRA HARDWARE AUDIO ROUTER READY               |")
        print("+==================================================================+")
        in_tag = "  [HEADSET MIC ACTIVE]" if is_headphone_mic else "  [LAPTOP BUILT-IN MIC]"
        out_tag = "  [HEADPHONES ACTIVE]" if is_headphones else "  [LAPTOP SPEAKERS ACTIVE]"
        print(f"  INPUT MIC  : [{best_in_idx}] {best_in_name}{in_tag}")
        print(f"  OUTPUT     : [{best_out_idx}] {best_out_name}{out_tag}")
        if vb_cable_idx is not None:
            print(f"  VSEEFACE   : [{vb_cable_idx}] {devs[vb_cable_idx]['name']}  [LIP-SYNC ACTIVE]")
        print("+==================================================================+\n")

    return config

def play_audio_dual(file_path: str, audio_config: dict = None):
    """
    Plays audio to the prioritized output device (Headphones/Speakers),
    and simultaneously routes audio to VB-Audio Cable for VSeeFace lip-sync if available.
    """
    if not file_path or not os.path.exists(file_path):
        return

    if audio_config is None:
        audio_config = detect_audio_devices(verbose=False)

    out_dev = audio_config.get('output_device')
    vb_dev = audio_config.get('vb_cable_device')

    try:
        data, fs = sf.read(file_path, dtype='float32')

        # If VB-Cable is available, play simultaneously in a background thread for VSeeFace lip-sync
        if vb_dev is not None and vb_dev != out_dev:
            def play_vb():
                try:
                    sd.play(data, fs, device=vb_dev)
                    sd.wait()
                except Exception:
                    pass
            threading.Thread(target=play_vb, daemon=True).start()

        # Primary playback directly to user's headphones or speakers
        sd.play(data, fs, device=out_dev)
        sd.wait()
    except Exception as e:
        print(f"--- [AUDIO ROUTER] Playback fallback: {e} ---")
        try:
            import winsound
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
        except Exception:
            pass
