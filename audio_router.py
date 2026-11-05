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
    """
    Tests if the input device can genuinely be opened by sounddevice
    AND verifies that the hardware stream actually captures non-zero audio samples.
    Rejects broken/silent drivers (such as Windows DirectSound on Intel SST arrays).
    """
    try:
        sd.check_input_settings(device=dev_idx, samplerate=fs, channels=channels)
        with sd.InputStream(device=dev_idx, samplerate=fs, channels=channels, blocksize=512, dtype='float32') as s:
            data, _ = s.read(512)
            if np.count_nonzero(data) == 0:
                data, _ = s.read(512)
                if np.count_nonzero(data) == 0:
                    return False
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

_cached_audio_config = None
_cached_audio_time = 0.0

def get_cached_audio_config(max_age_sec: float = 3.0, verbose: bool = False) -> dict:
    """Returns cached device configuration to avoid repeating 600ms portaudio device queries."""
    global _cached_audio_config, _cached_audio_time
    now = time.time()
    if _cached_audio_config is not None and (now - _cached_audio_time < max_age_sec):
        return _cached_audio_config
    _cached_audio_config = detect_audio_devices(verbose=verbose)
    _cached_audio_time = now
    return _cached_audio_config

def detect_audio_devices(verbose: bool = True) -> dict:
    """
    Scans all system audio devices, prioritizing native 48kHz WASAPI and DirectSound
    over legacy MME to eliminate sample rate conversion distortion and clicking.
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

    # Host API preference for OUTPUT: WASAPI (hostapi 2, native 48kHz float) > DirectSound (1) > MME (0)
    def _out_hostapi_priority(dev_idx):
        try:
            h = devs[dev_idx]['hostapi']
            if h == 2:
                return 0
            elif h == 1:
                return 1
            else:
                return 2
        except Exception:
            return 3

    # Host API preference for INPUT: MME (hostapi 0, Windows kernel mixer resamples to 16kHz) > DirectSound (1) > WASAPI (2)
    def _in_hostapi_priority(dev_idx):
        try:
            h = devs[dev_idx]['hostapi']
            if h == 0:
                return 0
            elif h == 1:
                return 1
            else:
                return 2
        except Exception:
            return 3

    out_device_indices = sorted(range(len(devs)), key=_out_hostapi_priority)
    in_device_indices = sorted(range(len(devs)), key=_in_hostapi_priority)

    best_out_idx = None
    best_out_name = None
    is_headphones = False

    best_in_idx = None
    best_in_name = None
    is_headphone_mic_active = False

    vb_cable_idx = None

    # 1. Detect VB-Cable for avatar lip-sync (prioritizing WASAPI)
    for i in out_device_indices:
        d = devs[i]
        if d['max_output_channels'] > 0 and is_valid_hostapi(d):
            nl = d['name'].lower()
            if 'cable input' in nl and can_open_output(i):
                vb_cable_idx = i
                break
    if vb_cable_idx is None:
        for i in out_device_indices:
            d = devs[i]
            if d['max_output_channels'] > 0 and is_valid_hostapi(d):
                nl = d['name'].lower()
                if ('cable in' in nl or 'virtual cable' in nl) and can_open_output(i):
                    vb_cable_idx = i
                    break

    # 2. Find Output: Priority 1 = Real Connected Headphones / Headset
    for i in out_device_indices:
        d = devs[i]
        if d['max_output_channels'] > 0 and is_valid_hostapi(d):
            if is_headphone_device(d['name']) and can_open_output(i):
                best_out_idx = i
                best_out_name = d['name']
                is_headphones = True
                break

    # Output: Priority 2 = Realtek / Physical Laptop Speakers (WASAPI prioritized)
    if best_out_idx is None:
        for i in out_device_indices:
            d = devs[i]
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
    for i in in_device_indices:
        d = devs[i]
        if d['max_input_channels'] > 0 and is_valid_hostapi(d):
            if is_headset_mic(d['name']) and can_open_input(i):
                best_in_idx = i
                best_in_name = d['name']
                is_headphone_mic_active = True
                break

    # Input: Priority 2 = Physical Built-in Microphone Array
    if best_in_idx is None:
        for i in in_device_indices:
            d = devs[i]
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

def is_vb_cable_listen_enabled() -> bool:
    """
    Checks Windows registry to see if 'Listen to this device' is enabled
    on CABLE Output (which loops VB-Cable directly into the speakers).
    """
    try:
        import winreg
        key_path = r"SOFTWARE\Microsoft\Windows\CurrentVersion\MMDevices\Audio\Capture"
        access = winreg.KEY_READ | winreg.KEY_WOW64_64KEY
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key_path, 0, access)
        num_keys = winreg.QueryInfoKey(root)[0]
        for i in range(num_keys):
            guid = winreg.EnumKey(root, i)
            try:
                pk = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, f"{key_path}\\{guid}\\Properties", 0, access)
                dev_name = ""
                listen_val = None
                for j in range(winreg.QueryInfoKey(pk)[1]):
                    k, v, _ = winreg.EnumValue(pk, j)
                    if k.endswith("{a45c254e-df1c-4efd-8020-67d146a850e0},2") or k.endswith("{b3f8fa53-0004-438e-9003-51a46e139bfc},6"):
                        dev_name = v
                    if "24dbb0fc-9311-4b3d-9cf0-18ff155639d4},1" in k:
                        listen_val = v
                winreg.CloseKey(pk)
                if dev_name and any(x in dev_name.lower() for x in ["cable", "vb-audio"]):
                    if listen_val and b'\xff\xff' in listen_val:
                        return True
            except Exception:
                pass
        winreg.CloseKey(root)
    except Exception:
        pass
    return False

def play_audio_dual(file_path: str, audio_config: dict = None):
    """
    Plays audio to the prioritized output device following the strict user policy:
    1. First check if headphones are connected.
    2. If YES (headphones connected): Speak in HEADPHONES ONLY.
       (Never route audio to laptop speakers or secondary virtual streams that bleed into laptop speakers).
    3. If NO (headphones not connected):
       - If Windows 'Listen to this device' is enabled on VB-Cable:
         Stream exclusively to VB-Cable (VB-Cable drives VSeeFace lip-sync, and Windows forwards it to Speakers with ZERO double voice).
       - Otherwise, stream to Speakers (+ VB-Cable if available).
    Includes multi-layer fallback to default audio device and winsound so words are never lost.
    """
    if not file_path or not os.path.exists(file_path):
        return

    # Use cached audio config (max 3s age) to eliminate the 600ms re-query lag on every utterance
    if audio_config is None:
        audio_config = get_cached_audio_config(max_age_sec=3.0, verbose=False)

    out_dev = audio_config.get('output_device')
    out_name = audio_config.get('output_name', 'Default Device')
    is_hp = audio_config.get('is_headphones', False)
    vb_dev = audio_config.get('vb_cable_device')

    vb_listen_active = is_vb_cable_listen_enabled()

    if is_hp:
        print(f"[AUDIO] Headphones Connected: Outputting speech exclusively to '{out_name}'.")
        target_devs = [out_dev]
    else:
        if vb_dev is not None and vb_listen_active:
            print(f"[AUDIO] Laptop Speakers + VSeeFace Lip-Sync Active: Single stream via VB-Cable (eliminates double voice).")
            target_devs = [vb_dev]
        elif vb_dev is not None:
            print(f"[AUDIO] Laptop Speakers '{out_name}' + VB-Cable Lip-Sync Active.")
            target_devs = [out_dev, vb_dev]
        else:
            print(f"[AUDIO] Headphones Not Detected: Outputting speech to Laptop Speakers '{out_name}'.")
            target_devs = [out_dev]

    played_successfully = False

    try:
        data, fs = sf.read(file_path, dtype='float32')

        # Add 300ms safety tail silence buffer to guarantee DAC / Bluetooth hardware flush
        tail_silence = np.zeros(int(fs * 0.30), dtype=np.float32)
        if data.ndim == 1:
            data = np.concatenate([data, tail_silence])
        else:
            tail_silence = np.zeros((int(fs * 0.30), data.shape[1]), dtype=np.float32)
            data = np.concatenate([data, tail_silence], axis=0)

        # Convert mono to stereo (2 channels) for clean hardware rendering with zero phase noise
        if data.ndim == 1:
            stereo_data = np.column_stack([data, data])
        else:
            stereo_data = data

        valid_devs = [d for d in target_devs if d is not None]

        if len(valid_devs) == 1:
            d_idx = valid_devs[0]
            try:
                sd.play(stereo_data, samplerate=fs, device=d_idx)
                sd.wait()
                played_successfully = True
            except Exception as e_play:
                print(f"--- [AUDIO ROUTER] Native sd.play notice on [{d_idx}]: {e_play} ---")
        elif len(valid_devs) > 1:
            # Multi-device synchronized parallel streaming (zero GIL jitter, zero buffer underflow)
            def _play_worker(dev_id):
                try:
                    sd.play(stereo_data, samplerate=fs, device=dev_id)
                    sd.wait()
                except Exception as ep:
                    print(f"--- [AUDIO ROUTER] Stream worker [{dev_id}] notice: {ep} ---")

            threads = [threading.Thread(target=_play_worker, args=(d,), daemon=True) for d in valid_devs]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            played_successfully = True

        time.sleep(0.10)

    except Exception as e:
        print(f"--- [AUDIO ROUTER] SoundDevice playback notice: {e} ---")

    # Final guaranteed fallback: Windows Multimedia Subsystem (winsound)
    if not played_successfully:
        try:
            print("--- [AUDIO ROUTER] Triggering Windows Multimedia fallback (winsound)... ---")
            winsound.PlaySound(file_path, winsound.SND_FILENAME)
        except Exception as we:
            print(f"--- [AUDIO ROUTER] Winsound fallback notice: {we} ---")

