import sounddevice as sd

print(sd.query_devices())
print("Default device:", sd.default.device)