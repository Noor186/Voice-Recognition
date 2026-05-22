import sounddevice as sd
import soundfile as sf
import os

SR = 16000
SECONDS = 4

person = input("Enter person name (must match exactly, e.g. Noor): ").strip()
os.makedirs(f"wav_files/{person}", exist_ok=True)

count = int(input("How many new samples to record? (recommend 10 per person): "))

for i in range(1, count + 1):
    # Count existing files fresh each iteration to avoid overwriting
    existing = len([f for f in os.listdir(f"wav_files/{person}") if f.endswith(".wav")])
    
    input(f"\n  Sample {i}/{count} — press Enter, then speak for {SECONDS} seconds...")
    audio = sd.rec(int(SECONDS * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    
    path = f"wav_files/{person}/{person}_{existing + 1:02d}.wav"
    sf.write(path, audio, SR)
    print(f"  ✔️ Saved: {path}")

print("\nAll done! Now re-run steps 2 and 3 to retrain.")
