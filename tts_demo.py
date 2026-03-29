import sys
sys.path.insert(0, 'C:/Users/hash_/VibeVoice/.claude/worktrees/vibrant-lamport')

from src.tts.streaming_bridge import VibeVoiceTTSService
import numpy as np
import wave

svc = VibeVoiceTTSService(
    model_path='C:/Users/hash_/.cache/huggingface/hub/models--microsoft--VibeVoice-Realtime-0.5B/snapshots/6bce5f06044837fe6d2c5d7a71a84f0416bd57e4',
    voices_dir='C:/Users/hash_/VibeVoice/.claude/worktrees/vibrant-lamport/demo/voices/streaming_model',
    device='cpu',
    inference_steps=5,
)
svc.load()

# ← CHANGE THIS TEXT
text = "Bucket in front of you, move left"

print('Generating audio for:', text)
all_chunks = []
for chunk in svc.stream(text):
    all_chunks.append(chunk)
    print(f'  got chunk: {chunk.shape[0]} samples')

# Save to WAV file
audio = np.concatenate(all_chunks)
audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)
out_path = 'tts_output.wav'
with wave.open(out_path, 'w') as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(audio_int16.tobytes())

print(f'Saved to {out_path} ({len(audio)/24000:.2f}s)')

# Try playing
try:
    import sounddevice as sd
    print('Playing via sounddevice...')
    sd.play(audio, samplerate=24000)
    sd.wait()
    print('Done.')
except Exception as e:
    print(f'sounddevice playback failed: {e}')
    print(f'Open {out_path} manually to hear the audio.')
