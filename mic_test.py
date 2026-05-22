import numpy as np
import librosa
import pickle
import sounddevice as sd

# ── MFCC config — must match 2_extract_features.py exactly ──────────────────
SR        = 16000
N_MFCC    = 13
N_FFT     = 512
HOP_LEN   = 160
WIN_LEN   = 400
TOP_DB    = 20
# ─────────────────────────────────────────────────────────────────────────────

RECORD_SECONDS = 4

# Load model + scaler
bundle = pickle.load(open("model.pkl", "rb"))
model  = bundle["model"]
scaler = bundle["scaler"]

def extract_features(audio):
    # Remove DC offset (mic bias)
    audio = audio - np.mean(audio)

    # Check raw loudness BEFORE normalizing
    if np.max(np.abs(audio)) < 0.01:
        return None

    # Normalize
    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    # Trim silence — this is the key fix for live mic
    audio, _ = librosa.effects.trim(audio, top_db=TOP_DB)

    if len(audio) == 0:
        return None

    mfcc = librosa.feature.mfcc(
        y=audio, sr=SR,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        win_length=WIN_LEN
    )

    return np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])


print(f"Recording for {RECORD_SECONDS} seconds... speak now 🎙")

recording = sd.rec(int(RECORD_SECONDS * SR), samplerate=SR, channels=1, dtype="float32")
sd.wait()

audio = recording.flatten()

features = extract_features(audio)

if features is None:
    print("Audio too weak or silent — please speak louder or closer to the mic.")
else:
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    proba = model.predict_proba(features_scaled)
    confidence = np.max(proba)
    print(f"Predicted speaker : {prediction[0]}")
    print(f"Confidence        : {confidence:.2%}")