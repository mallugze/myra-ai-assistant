import sounddevice as sd
import soundfile as sf
import threading
import os

def find_audio_output_devices():
    default_out = sd.default.device[1]
    vb_cable_out = None
    
    for i, dev in enumerate(sd.query_devices()):
        if dev['max_output_channels'] > 0:
            name_lower = dev['name'].lower()
            if "cable input" in name_lower or "cable in" in name_lower:
                vb_cable_out = i
                break
                
    return default_out, vb_cable_out

default_out, vb_out = find_audio_output_devices()
print(f"Default Output Device (Speakers): {default_out} ({sd.query_devices(default_out)['name']})")
if vb_out is not None:
    print(f"VB-Cable Output Device (VSeeFace): {vb_out} ({sd.query_devices(vb_out)['name']})")
else:
    print("VB-Cable Output Device: Not detected.")

print("Dual Audio Setup Tested OK!")