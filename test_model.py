import numpy as np
import librosa
import pickle

# ── MFCC config — must match 2_extract_features.py exactly ──────────────────
SR        = 16000
N_MFCC    = 13
N_FFT     = 512
HOP_LEN   = 160
WIN_LEN   = 400
TOP_DB    = 20
# ─────────────────────────────────────────────────────────────────────────────

# Load model + scaler
bundle = pickle.load(open("model.pkl", "rb"))
model  = bundle["model"]
scaler = bundle["scaler"]

def extract_features(file_path):
    audio, sr = librosa.load(file_path, sr=SR)
    audio, _ = librosa.effects.trim(audio, top_db=TOP_DB)

    if len(audio) == 0:
        return None

    if np.max(np.abs(audio)) < 0.01:
        return None

    audio = audio / (np.max(np.abs(audio)) + 1e-9)

    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr,
        n_mfcc=N_MFCC,
        n_fft=N_FFT,
        hop_length=HOP_LEN,
        win_length=WIN_LEN
    )

    return np.hstack([mfcc.mean(axis=1), mfcc.std(axis=1)])


# ── Change this path to test a different file ────────────────────────────────
test_file = "wav_files/Noor/noor_02.wav"
# ─────────────────────────────────────────────────────────────────────────────

features = extract_features(test_file)

if features is None:
    print("Could not process audio (silent or empty)")
else:
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)
    proba = model.predict_proba(features_scaled)
    confidence = np.max(proba)
    print(f"Predicted speaker : {prediction[0]}")
    print(f"Confidence        : {confidence:.2%}")