import numpy as np
import pickle
import collections
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix

# ─── Load Dataset ────────────────────────────────────────────────
data = np.load("dataset.npz")
X = data["X"]
y = np.array(data["y"], dtype=str)  # fix np.str_ issue

print(f"Loaded {X.shape[0]} samples, {X.shape[1]} features each")

# ─── Sample Count Per Speaker ────────────────────────────────────
print("\nSamples per speaker:")
for speaker, count in sorted(collections.Counter(y).items()):
    print(f"  {speaker}: {count} samples")

# ─── Scale Features ──────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ─── Train Final Model ───────────────────────────────────────────
model = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
model.fit(X_scaled, y)

# ─── Accuracy ────────────────────────────────────────────────────
scores = cross_val_score(model, X_scaled, y, cv=5)
print(f"\nFinal accuracy: {scores.mean():.2%} ± {scores.std():.2%}")

# ─── Confusion Matrix ────────────────────────────────────────────
labels = sorted(set(y))
y_pred = cross_val_predict(model, X_scaled, y, cv=5)
cm = confusion_matrix(y, y_pred, labels=labels)

print("\nConfusion Matrix:")
print(f"{'':15}", end="")
for label in labels:
    print(f"{label[:8]:>10}", end="")
print()
for i, label in enumerate(labels):
    print(f"{label[:15]:15}", end="")
    for val in cm[i]:
        print(f"{val:>10}", end="")
    print()

# ─── Save Model ──────────────────────────────────────────────────
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "scaler": scaler}, f)

print("\nModel and scaler saved to model.pkl ✔️")