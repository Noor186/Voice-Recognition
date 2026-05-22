import os
import numpy as np
import librosa

# ── MFCC config ──────────────────────────────────────────────────
SR      = 16000
N_MFCC  = 13
N_FFT   = 512
HOP_LEN = 160
WIN_LEN = 400
TOP_DB  = 20
# ─────────────────────────────────────────────────────────────────

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


X = []
y = []
file_names = []  # track filenames alongside features

data_path = "wav_files"

for person in os.listdir(data_path):
    person_path = os.path.join(data_path, person)
    if not os.path.isdir(person_path):
        continue

    person_samples = 0
    for file in os.listdir(person_path):
        if not file.endswith(".wav"):
            continue

        file_path = os.path.join(person_path, file)
        features = extract_features(file_path)

        if features is not None:
            X.append(features)
            y.append(person)
            file_names.append(file_path)
            person_samples += 1
        else:
            print(f"Skipped (silent/empty): {file_path}")

    print(f"{person}: {person_samples} samples loaded")

X = np.array(X)
y = np.array(y)

# ── Find Outlier Files Per Person ────────────────────────────────
print("\n── Outlier Detection ───────────────────────────────────────")
for person in sorted(set(y)):
    idx = np.where(y == person)[0]
    person_X = X[idx]
    person_files = [file_names[i] for i in idx]

    # Distance of each sample from the person's mean
    mean_vec = person_X.mean(axis=0)
    distances = np.linalg.norm(person_X - mean_vec, axis=1)

    # Flag anything more than 1.5 std away from mean distance
    dist_mean = distances.mean()
    dist_std  = distances.std()
    threshold = dist_mean + 1.5 * dist_std

    outliers = [(person_files[i], distances[i]) 
                for i in range(len(distances)) 
                if distances[i] > threshold]

    if outliers:
        print(f"\n{person} — suspicious files:")
        for fname, dist in sorted(outliers, key=lambda x: -x[1]):
            print(f"  ⚠️  {os.path.basename(fname)}  (score: {dist:.2f})")
    else:
        print(f"{person} — all samples look consistent ✅")

# ── Save Dataset ─────────────────────────────────────────────────
np.savez("dataset.npz", X=X, y=y)
print(f"\nDone ✔️  Total samples: {X.shape[0]}, Feature size: {X.shape[1]}")