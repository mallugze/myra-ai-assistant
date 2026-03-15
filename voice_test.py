import torch
import soundfile as sf

device = torch.device("cpu")


model, example_text = torch.hub.load(
    repo_or_dir='snakers4/silero-models',
    model='silero_tts',
    language='en',
    speaker='v3_en'
)



text = "Hmm… tired already? That didn't take long."


audio = model.apply_tts(
    text=text,
    speaker='en_0',
    sample_rate=48000
)

sf.write("silero_test.wav", audio, 48000)

print("Audio saved as silero_test.wav")
