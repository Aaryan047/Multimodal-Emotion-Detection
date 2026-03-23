import os
import numpy as np
import pickle
import nibabel as nib
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# =========================
# PATHS (GITHUB FRIENDLY)
# =========================

BASE = os.path.dirname(os.path.abspath(__file__))

speech_path = os.path.join(BASE, "Speech")
eeg_path = os.path.join(BASE, "EEG")
mri2_path = os.path.join(BASE, "MRI2")

# =========================
# SPEECH FEATURES
# =========================

def extract_audio_features(file_path):
    sr, audio = wav.read(file_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    return [np.mean(audio), np.std(audio), np.max(audio), np.min(audio)]

def get_emotion(filename):
    emotion_dict = {
        '01': 'neutral',
        '02': 'neutral',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'sad',
        '07': 'angry',
        '08': 'happy'
    }
    
    emo = emotion_dict[filename.split('-')[2]]
    return "happy" if emo == "happy" else "sad"

X_speech, y_speech = [], []

actors = sorted(os.listdir(speech_path))[:24]

for actor in actors:
    actor_path = os.path.join(speech_path, actor)
    for file in os.listdir(actor_path):
        try:
            X_speech.append(extract_audio_features(os.path.join(actor_path, file)))
            y_speech.append(get_emotion(file))
        except:
            continue

# =========================
# EEG FEATURES
# =========================

def valence_to_emotion(v):
    return "happy" if v > 5 else "sad"

X_eeg, y_eeg = [], []

for i in range(1, 33):
    file = f"s{i:02d}.dat"
    file_path = os.path.join(eeg_path, file)
    
    with open(file_path, "rb") as f:
        data = pickle.load(f, encoding='latin1')
    
    eeg = data['data']
    labels = data['labels']
    
    for j in range(len(eeg)):
        trial = eeg[j]
        X_eeg.append([
            np.mean(trial),
            np.std(trial),
            np.max(trial),
            np.min(trial)
        ])
        y_eeg.append(valence_to_emotion(labels[j][0]))

# =========================
# MRI (NeuroEmo)
# =========================

X_mri, y_mri = [], []

for subject in os.listdir(mri2_path):
    func_path = os.path.join(mri2_path, subject, "func")

    if not os.path.exists(func_path):
        continue

    for file in os.listdir(func_path):
        if "bold" in file and (file.endswith(".nii") or file.endswith(".nii.gz")):
            try:
                path = os.path.join(func_path, file)
                img = nib.load(path)
                data = np.asarray(img.dataobj)

                # fMRI is 4D → take middle slice + time
                slice_2d = data[:, :, data.shape[2]//2, data.shape[3]//2]

                features = [
                    np.mean(slice_2d),
                    np.std(slice_2d),
                    np.max(slice_2d),
                    np.min(slice_2d)
                ]

                X_mri.append(features)

                # Subject-based labeling
                subject_num = int(subject.split('-')[1])
                label = "happy" if subject_num % 2 == 0 else "sad"

                y_mri.append(label)

            except Exception as e:
                print(f"Skipping {file} due to error:", e)

# =========================
# TRAIN MODELS
# =========================

# Speech
X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
    X_speech, y_speech, test_size=0.2, random_state=42
)

speech_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=60
)
speech_model.fit(X_s_train, y_s_train)

# EEG
X_e_train, X_e_test, y_e_train, y_e_test = train_test_split(
    X_eeg, y_eeg, test_size=0.2, random_state=42
)

eeg_model = SVC(kernel='rbf', probability=True)
eeg_model.fit(X_e_train, y_e_train)

# MRI
X_m_train, X_m_test, y_m_train, y_m_test = train_test_split(
    X_mri, y_mri, test_size=0.2, random_state=42
)

mri_model = SVC(kernel='rbf', probability=True)
mri_model.fit(X_m_train, y_m_train)

# =========================
# PREDICTIONS
# =========================

speech_probs = speech_model.predict_proba(X_s_test)
eeg_probs = eeg_model.predict_proba(X_e_test)
mri_probs = mri_model.predict_proba(X_m_test)

# align sizes
n = min(len(speech_probs), len(eeg_probs), len(mri_probs))

speech_probs = speech_probs[:n]
eeg_probs = eeg_probs[:n]
mri_probs = mri_probs[:n]

classes = speech_model.classes_

# =========================
# FUSION
# =========================

weights = {
    "EEG": 0.3,
    "Speech": 0.4,
    "MRI": 0.3
}

fusion_preds = []

for i in range(n):
    combined = (
        weights["Speech"] * speech_probs[i] +
        weights["EEG"] * eeg_probs[i] +
        weights["MRI"] * mri_probs[i]
    )
    
    fusion_preds.append(classes[np.argmax(combined)])

fusion_true = y_s_test[:n]

# =========================
# ACCURACY
# =========================

speech_acc = accuracy_score(y_s_test, speech_model.predict(X_s_test))
eeg_acc = accuracy_score(y_e_test, eeg_model.predict(X_e_test))
mri_acc = accuracy_score(y_m_test, mri_model.predict(X_m_test))
fusion_acc = accuracy_score(fusion_true, fusion_preds)

print("\n===== FINAL RESULTS =====")
print("Speech Accuracy:", round(speech_acc * 100, 2), "%")
print("EEG Accuracy:", round(eeg_acc * 100, 2), "%")
print("MRI Accuracy:", round(mri_acc * 100, 2), "%")
print("Fusion Accuracy:", round(fusion_acc * 100, 2), "%")

# =========================
# PLOT
# =========================

models = ['Speech', 'EEG', 'MRI', 'Fusion']
accuracies = [
    speech_acc * 100,
    eeg_acc * 100,
    mri_acc * 100,
    fusion_acc * 100
]

plt.figure()
plt.bar(models, accuracies)
plt.xlabel("Models")
plt.ylabel("Accuracy (%)")
plt.title("Multimodal Emotion Recognition")
plt.show()

# =========================
# SAFE SAVE BLOCK
# =========================

try:
    import joblib

    model_path = os.path.join(BASE, "saved_models")

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Save models
    joblib.dump(speech_model, os.path.join(model_path, "speech_model.pkl"))
    joblib.dump(eeg_model, os.path.join(model_path, "eeg_model.pkl"))
    joblib.dump(mri_model, os.path.join(model_path, "mri_model.pkl"))

    # Save fusion config
    fusion_config = {
        "weights": weights,
        "classes": list(classes)
    }

    joblib.dump(fusion_config, os.path.join(model_path, "fusion_config.pkl"))

    # Save results
    results = {
        "Speech Accuracy": speech_acc,
        "EEG Accuracy": eeg_acc,
        "MRI Accuracy": mri_acc,
        "Fusion Accuracy": fusion_acc
    }

    joblib.dump(results, os.path.join(model_path, "results.pkl"))

    print("\nModels saved successfully!")

except Exception as e:
    print("\nSaving failed:", e)
