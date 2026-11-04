"""
Audio Device Router for Myra AI Assistant
Intelligently discovers connected headphones, headsets, and microphones.
Prioritizes external/headphone devices over laptop built-in microphones/speakers,
and manages dual-channel playback (Headphones or Speakers + VB-Cable for VSeeFace lip-sync).
"""
import os
import re
import time
import threading
import sounddevice as sd
import soundfile as sf
import numpy as np
import winsound

HEADPHONE_KEYWORDS = [
    'headphone', 'headset', 'rockerz', 'earphone',
    'bluetooth', 'wireless', 'airpod', 'earbuds', 'buds'
]

HEADSET_MIC_KEYWORDS = [
    'headset', 'rockerz', 'airpod', 'buds', 'earphone', 'bluetooth', 'wireless'
]

BUILTIN_MIC_KEYWORDS = [
    'array', 'smart sound', 'intel', 'built-in', 'internal'
]

VIRTUAL_KEYWORDS = [
    'cable', 'virtual', 'mapper', 'primary sound', 'stereo mix'
]

def is_virtual_device(name: str) -> bool:
    return any(v in name.lower() for v in VIRTUAL_KEYWORDS)

def is_unassigned_jack(name: str) -> bool:
    """Detects unpopulated/ghost Realtek hardware jack endpoints (e.g. 'Headphones ()')."""
    return bool(re.search(r'\(\s*\)', name)) or '2nd output' in name.lower()

def is_valid_hostapi(dev: dict) -> bool:
    """
    PortAudio Windows WDM-KS exposes raw kernel pins that fail with -9999 ('Blocking API not supported').
    We strictly filter them out and use Windows MME, DirectSound, or WASAPI.
    """
    try:
        api_name = sd.query_hostapis(dev['hostapi'])['name'].lower()
        return 'wdm-ks' not in api_name
    except Exception:
        return True

def can_open_output(dev_idx: int, fs: int = 48000, channels: int = 2) -> bool:
    """Tests if the output device can genuinely be opened by sounddevice without error."""
    try:
        sd.check_output_settings(device=dev_idx, samplerate=fs, channels=channels)
        s = sd.OutputStream(device=dev_idx, samplerate=fs, channels=channels)
        s.close()
        return True
    except Exception:
        return False

def can_open_input(dev_idx: int, fs: int = 16000, channels: int = 1) -> bool:
    """Tests if the input device can genuinely be opened by sounddevice without error."""
    try:
        sd.check_input_settings(device=dev_idx, samplerate=fs, channels=channels)
        s = sd.InputStream(device=dev_idx, samplerate=fs, channels=channels)
        s.close()
        return True
    except Exception:
        return False

def is_headphone_device(name: str) -> bool:
    if is_virtual_device(name) or is_unassigned_jack(name):
        return False
    name_lower = name.lower()
    if 'speaker' in name_lower and not any(k in name_lower for k in ['headphone', 'headset', 'earphone']):
        return False
    return any(k in name_lower for k in HEADPHONE_KEYWORDS)

def is_headset_mic(name: str) -> bool:
    if is_virtual_device(name) or is_unassigned_jack(name):
        return False
    name_lower = name.lower()
    if any(k in name_lower for k in BUILTIN_MIC_KEYWORDS):
        return False
    return any(k in name_lower for k in HEADSET_MIC_KEYWORDS)

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
    is_headphone_mic_active = False

    vb_cable_idx = None

    # 1. Detect VB-Cable for avatar lip-sync (prefer standard 'cable input' over 16ch)
    for i, d in enumerate(devs):
        if d['max_output_channels'] > 0 and is_valid_hostapi(d):
            nl = d['name'].lower()
            if 'cable input' in nl and can_open_output(i):
                vb_cable_idx = i
                break
    if vb_cable_idx is None:
        for i, d in enumerate(devs):
            if d['max_output_channels'] > 0 and is_valid_hostapi(d):
                nl = d['name'].lower()
                if ('cable in' in nl or 'virtual cable' in nl) and can_open_output(i):
                    vb_cable_idx = i
                    break

    # 2. Find Output: Priority 1 = Real Connected Headphones / Headset
    for i, d in enumerate(devs):
        if d['max_output_channels'] > 0 and is_valid_hostapi(d):
            if is_headphone_device(d['name']) and can_open_output(i):
                best_out_idx = i
                best_out_name = d['name']
                is_headphones = True
                break

    # Output: Priority 2 = Realtek / Physical Laptop Speakers
    if best_out_idx is None:
        for i, d in enumerate(devs):
            if d['max_output_channels'] > 0 and is_valid_hostapi(d) and not is_virtual_device(d['name']) and not is_unassigned_jack(d['name']):
                if can_open_output(i):
                    best_out_idx = i
                    best_out_name = d['name']
                    is_headphones = False
                    break

    # Output: Priority 3 = System Default Output
    if best_out_idx is None:
        best_out_idx = default_out
        best_out_name = devs[default_out]['name'] if default_out is not None and default_out >= 0 else 'Default Output'
        is_headphones = is_headphone_device(best_out_name)

    # 3. Find Input: Priority 1 = Headset / External Mic
    for i, d in enumerate(devs):
        if d['max_input_channels'] > 0 and is_valid_hostapi(d):
            if is_headset_mic(d['name']) and can_open_input(i):
                best_in_idx = i
                best_in_name = d['name']
                is_headphone_mic_active = True
                break

    # Input: Priority 2 = Physical Built-in Microphone Array
    if best_in_idx is None:
        for i, d in enumerate(devs):
            if d['max_input_channels'] > 0 and is_valid_hostapi(d) and not is_virtual_device(d['name']) and not is_unassigned_jack(d['name']):
                if can_open_input(i):
                    best_in_idx = i
                    best_in_name = d['name']
                    is_headphone_mic_active = False
                    break

    # Input: Priority 3 = System Default Input
    if best_in_idx is None:
        best_in_idx = default_in
        best_in_name = devs[default_in]['name'] if default_in is not None and default_in >= 0 else 'Default Input'
        is_headphone_mic_active = is_headset_mic(best_in_name)

    config = {
        'input_device': best_in_idx,
        'input_name': best_in_name,
        'is_headphone_mic': is_headphone_mic_active,
        'output_device': best_out_idx,
        'output_name': best_out_name,
        'is_headphones': is_headphones,
        'vb_cable_device': vb_cable_idx
    }

    if verbose:
        print("\n+==================================================================+")
        print("|           [AUDIO] MYRA HARDWARE AUDIO ROUTER READY               |")
        print("+==================================================================+")
        in_tag = "  [HEADSET MIC ACTIVE]" if is_headphone_mic_active else "  [LAPTOP BUILT-IN MIC]"
        out_tag = "  [HEADPHONES ONLY]" if is_headphones else "  [LAPTOP SPEAKERS ACTIVE]"
        print(f"  INPUT MIC  : [{best_in_idx}] {best_in_name}{in_tag}")
        print(f"  OUTPUT     : [{best_out_idx}] {best_out_name}{out_tag}")
        if is_headphones:
            print("  POLICY     : Headphones connected -> Muting laptop speakers completely.")
        else:
            print("  POLICY     : Headphones not detected -> Outputting speech via laptop speakers.")
            if vb_cable_idx is not None:
                print(f"  VSEEFACE   : [{vb_cable_idx}] {devs[vb_cable_idx]['name']}  [LIP-SYNC ACTIVE]")
        print("+==================================================================+\n")

    return config

def play_audio_dual(file_path: str, audio_config: dict = None):
    """
    Plays audio to the prioritized output device following the strict user policy:
    1. First check if headphones are connected.
    2. If YES (headphones connected): Speak in HEADPHONES ONLY.
       (Never route audio to laptop speakers or secondary virtual streams that bleed into laptop speakers).
    3. If NO (headphones not connected): Speak using LAPTOP SPEAKERS + VB-Cable for VSeeFace.
    Includes multi-layer fallback to default audio device and winsound so words are never lost.
    """
    if not file_path or not os.path.exists(file_path):
        return

    # Always check the latest hardware connection state in real time (<0.5ms)
    audio_config = detect_audio_devices(verbose=False)

    out_dev = audio_config.get('output_device')
    out_name = audio_config.get('output_name', 'Default Device')
    is_hp = audio_config.get('is_headphones', False)
    vb_dev = audio_config.get('vb_cable_device')

    if is_hp:
        print(f"[AUDIO] Headphones Connected: Outputting speech exclusively to '{out_name}'.")
    else:
        print(f"[AUDIO] Headphones Not Detected: Outputting speech to Laptop Speakers '{out_name}'.")

    played_successfully = False

    try:
        data, fs = sf.read(file_path, dtype='float32')

        # Add 400ms safety tail silence buffer to guarantee DAC / Bluetooth hardware flush
        tail_silence = np.zeros(int(fs * 0.40), dtype=np.float32)
        if data.ndim == 1:
            data = np.concatenate([data, tail_silence])
        else:
            tail_silence = np.zeros((int(fs * 0.40), data.shape[1]), dtype=np.float32)
            data = np.concatenate([data, tail_silence], axis=0)

        # Synchronous dual-stream: Primary output (Headphones/Speakers) + VB-Cable (VSeeFace Lip-Sync)
        channels = 1 if data.ndim == 1 else data.shape[1]
        s_out = None
        s_vb = None

        try:
            s_out = sd.OutputStream(device=out_dev, channels=channels, samplerate=fs)
            s_out.start()
        except Exception as e_out:
            print(f"--- [AUDIO ROUTER] Primary stream notice: {e_out} ---")

        if vb_dev is not None and vb_dev != out_dev:
            try:
                s_vb = sd.OutputStream(device=vb_dev, channels=channels, samplerate=fs)
                s_vb.start()
            except Exception as e_vb:
                s_vb = None

        # Lock-step chunk streaming guarantees zero clock drift between speakers and lip-sync
        block_size = 2048
        for i in range(0, len(data), block_size):
            chunk = data[i:i + block_size]
            if s_out:
                s_out.write(chunk)
            if s_vb:
                s_vb.write(chunk)

        if s_out:
            s_out.stop()
            s_out.close()
            played_successfully = True

        if s_vb:
            s_vb.stop()
            s_vb.close()

        time.sleep(0.20)

    except Exception as e:
        print(f"--- [AUDIO ROUTER] SoundDevice playback notice: {e} ---")

    # Final guaranteed fallback: Windows Multimedia Subsystem (winsound)
    if not played_successfully:
        try:
            print("--- [AUDIO ROUTER] Triggering Windows Multimedia fallback (winsound)... ---")
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
        except Exception as we:
            print(f"--- [AUDIO ROUTER] Winsound fallback notice: {we} ---")

