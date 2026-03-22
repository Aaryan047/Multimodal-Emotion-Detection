# Multimodal-Emotion-Detection

## Overview

This project implements a multimodal emotion recognition system using three different data modalities:

* Speech (audio signals)
* EEG (brainwave signals)
* MRI (functional brain imaging)

The system combines predictions from individual models using a weighted decision-level fusion approach to improve overall performance.

The goal of this project is to explore how different modalities contribute to emotion recognition and to analyze the impact of model complexity versus feature representation.

---

## Problem Statement

Emotion recognition is a complex task that benefits from multiple sources of information. Single-modality systems (e.g., only speech or only EEG) often fail to capture the full context of human emotions.

This project addresses:

* How different modalities perform individually
* Whether combining them improves accuracy
* The limitations of each modality in practical scenarios

---

## Dataset Description

### 1. Speech Dataset (RAVDESS)

* Contains emotional speech recordings from multiple actors
* Audio files in `.wav` format
* Emotions mapped to binary classes:

  * Happy
  * Sad (includes neutral and negative emotions)

### 2. EEG Dataset (DEAP)

* Contains EEG recordings from 32 subjects
* Each trial includes brain activity signals and emotional labels
* Valence score used to classify:

  * Valence > 5 → Happy
  * Valence ≤ 5 → Sad

### 3. MRI Dataset (NeuroEmo)

* Functional MRI (fMRI) scans
* 4D brain imaging data (x, y, z, time)
* Data extracted from BOLD signals (functional activity)

Note:
The MRI dataset does not contain direct emotion labels for this setup. Proxy labels were generated based on subject grouping to enable supervised learning.

---

## Feature Extraction

### Speech Features

* Mean amplitude
* Standard deviation
* Maximum value
* Minimum value

These are simple time-domain statistical features extracted from raw audio signals.

### EEG Features

* Mean signal value
* Standard deviation
* Maximum value
* Minimum value

Computed per trial from EEG recordings.

### MRI Features

* Middle slice extracted from 4D fMRI data
* Features:

  * Mean intensity
  * Standard deviation
  * Maximum value
  * Minimum value

This reduces high-dimensional MRI data into a manageable feature representation.

---

## Models Used

| Modality | Model                        |
| -------- | ---------------------------- |
| Speech   | Random Forest                |
| EEG      | Support Vector Machine (SVM) |
| MRI      | Support Vector Machine (SVM) |

### Rationale

* **Random Forest (Speech):** Handles small feature sets well and captures non-linear relationships.
* **SVM (EEG):** Performs strongly on structured numerical data and small feature spaces.
* **SVM (MRI):** Used as a lightweight model due to limited feature representation.

---

## Fusion Strategy

A decision-level fusion approach is used.

Each model outputs class probabilities. These are combined using weighted averaging:

* EEG: 0.8 (Strong modality)
* Speech: 0.15 (Moderate modality)
* MRI: 0.05 (Weak modality)

Final prediction is obtained by selecting the class with the highest combined probability.

---

## Results

Typical performance observed:

* Speech Accuracy: ~72%
* EEG Accuracy: ~79%
* MRI Accuracy: ~50%
* Fusion Accuracy: ~87%

Key observation:
Fusion improves performance significantly by leveraging the strongest modality (EEG) while incorporating additional signals.

---

## Key Insights

1. **EEG is the strongest modality**

   * Directly captures brain activity
   * Provides the most reliable signal for emotion detection

2. **Speech provides moderate performance**

   * Limited by simple feature extraction
   * Could improve with MFCC or temporal features

3. **MRI contributes minimally**

   * High-dimensional data reduced to simple features
   * Lack of true emotion labels limits effectiveness

4. **Model complexity does not guarantee improvement**

   * Neural networks (MLP) did not outperform classical models
   * Feature quality is more important than model complexity

5. **Fusion effectiveness depends on weighting**

   * Poor weighting can degrade performance
   * Proper weighting improves overall accuracy

---

## Limitations

* MRI dataset lacks explicit emotion labels
* Feature extraction is simplistic (no spectral or temporal features)
* No synchronization between modalities
* Deep learning models are underutilized due to limited feature representation

---

## Future Work

* Use MFCC or spectrograms for speech
* Apply LSTM for sequential modeling
* Use CNN for MRI slices
* Incorporate properly labeled multimodal datasets
* Implement feature-level fusion instead of decision-level fusion

---

## Project Structure

```
Emotion_Detection/
│
├── Model_Final.py
├── README.md
├── .gitignore
```

Datasets and saved models are excluded from the repository.

---

## How to Run

1. Place datasets in the project directory:

   * Speech/
   * EEG/
   * MRI2/

2. Install dependencies:

```
pip install numpy scikit-learn matplotlib nibabel
```

3. Run:

```
python Model_Final.py
```

---

## Conclusion

This project demonstrates a complete multimodal emotion recognition pipeline using classical machine learning techniques and decision-level fusion.

The results highlight the importance of:

* Data quality
* Feature engineering
* Modality selection

over simply increasing model complexity.
