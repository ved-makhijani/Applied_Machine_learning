# COS30082 — Face Recognition Attendance System

Enterprise attendance system with face verification, emotion detection,
anti-spoofing (liveness), and a criminal watchlist (innovation feature).

## Components
- **Face verification** — MobileNetV2 embeddings, two approaches compared:
  classification (softmax) and metric learning (triplet loss).
  Trained on a 500-identity subset of the 11-785 HW2P2 dataset (4000 identities).
- **Emotion detection** — MobileNetV2 on FER2013, 7 classes, two-stage transfer learning.
- **Anti-spoofing** — MobileNetV2 binary liveness classifier (real/fake faces).
- **Criminal watchlist** — cosine-similarity screening reusing the face embeddings.

## Results (official verification_pairs_val.txt — 8805 pairs)
| Approach        | Cosine AUC | Euclidean AUC |
|-----------------|-----------|---------------|
| Classification  | 0.7827    | 0.7827        |
| Triplet         | 0.7935    | 0.7935        |

Triplet (metric learning) outperforms classification, as expected for open-set
verification. Cosine and Euclidean are identical because embeddings are L2-normalized.

## Structure
- `models/` — trained model weights
- `results/` — ROC curves, confusion matrices, metrics JSON
- `notebooks/` — training + evaluation notebook
- `ui/` — web frontend + Flask backend (if included)

## Datasets (not included — download separately)
- Face: 11-785 Fall'20 HW2P2 (Kaggle competition)
- Emotion: FER2013 (kaggle.com/datasets/msambare/fer2013)
- Liveness: Real and Fake Face Detection (kaggle.com/datasets/ciplab/real-and-fake-face-detection)
