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
import numpy as np

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
        'vb_cable_device': None if is_headphones else vb_cable_idx
    }

    if verbose:
        print("\n+==================================================================+")
        print("|           [AUDIO] MYRA HARDWARE AUDIO ROUTER READY               |")
        print("+==================================================================+")
        in_tag = "  [HEADSET MIC ACTIVE]" if is_headphone_mic else "  [LAPTOP BUILT-IN MIC]"
        out_tag = "  [HEADPHONES ONLY]" if is_headphones else "  [LAPTOP SPEAKERS ACTIVE]"
        print(f"  INPUT MIC  : [{best_in_idx}] {best_in_name}{in_tag}")
        print(f"  OUTPUT     : [{best_out_idx}] {best_out_name}{out_tag}")
        if is_headphones:
            print("  POLICY     : Headphones connected -> Muting laptop speakers completely.")
        elif vb_cable_idx is not None:
            print(f"  VSEEFACE   : [{vb_cable_idx}] {devs[vb_cable_idx]['name']}  [LIP-SYNC ACTIVE]")
        print("+==================================================================+\n")

    return config

def play_audio_dual(file_path: str, audio_config: dict = None):
    """
    Plays audio to the prioritized output device following the strict user policy:
    1. First check if headphones are connected.
    2. If YES (headphones connected): Speak in HEADPHONES ONLY.
       (Never route audio to laptop speakers or secondary virtual streams that bleed into laptop speakers).
    3. If NO (headphones not connected): Speak using LAPTOP SPEAKERS.
    Uses dedicated stream handles and hardware buffer drainage so words are never cut off.
    """
    if not file_path or not os.path.exists(file_path):
        return

    # Always check the latest hardware connection state in real time (<0.5ms)
    audio_config = detect_audio_devices(verbose=False)

    out_dev = audio_config.get('output_device')
    out_name = audio_config.get('output_name', 'Default Device')
    is_hp = audio_config.get('is_headphones', False)

    if is_hp:
        print(f"🎧 [AUDIO] Headphones Connected: Outputting speech exclusively to '{out_name}'.")
    else:
        print(f"🔊 [AUDIO] Headphones Not Detected: Outputting speech to Laptop Speakers '{out_name}'.")

    try:
        data, fs = sf.read(file_path, dtype='float32')

        # Add 400ms safety tail silence buffer to guarantee DAC / Bluetooth hardware flush
        tail_silence = np.zeros(int(fs * 0.40), dtype=np.float32)
        if data.ndim == 1:
            data = np.concatenate([data, tail_silence])
        else:
            tail_silence = np.zeros((int(fs * 0.40), data.shape[1]), dtype=np.float32)
            data = np.concatenate([data, tail_silence], axis=0)

        # Single dedicated stream: Headphones only if connected, otherwise Laptop Speakers
        try:
            s_out = sd.OutputStream(device=out_dev, channels=1 if data.ndim == 1 else data.shape[1], samplerate=fs)
            s_out.start()
            block_size = 2048
            for i in range(0, len(data), block_size):
                chunk = data[i:i + block_size]
                s_out.write(chunk)
            s_out.stop()
            s_out.close()
            # Safety sleep for physical Bluetooth headphone / speaker driver flush
            time.sleep(0.25)
        except Exception as se:
            sd.play(data, fs, device=out_dev)
            sd.wait()
            time.sleep(0.25)

    except Exception as e:
        print(f"--- [AUDIO ROUTER] Playback notice: {e} ---")
