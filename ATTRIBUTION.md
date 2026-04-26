# ATTRIBUTION — Pantanal BirdCLEF 2026

This document lists all external sources, datasets, pretrained models, libraries, and AI-generated content used in this project.

---

## Competition & Dataset

| Source | Details |
|---|---|
| **BirdCLEF 2026** | Kaggle competition. Data files: `train.csv`, `train_soundscapes_labels.csv`, `taxonomy.csv`, soundscape and bird-clip audio. URL: https://www.kaggle.com/competitions/birdclef-2026 |


---

## Pretrained Model

| Source | Details |
|---|---|
| **CNN14 (PANNs)** | Pretrained audio neural network backbone. Checkpoint: `Cnn14_mAP=0.431.pth`. From: Kong, Q., Cao, Y., Iqbal, T., Wang, Y., Wang, W., & Plumbley, M. D. (2020). *PANNs: Large-Scale Pretrained Audio Neural Networks for Audio Pattern Recognition.* IEEE/ACM TASLP. https://zenodo.org/record/3987831 |
| **audioset_tagging_cnn** | Repository containing CNN14 model code. Author: Qiuqiang Kong. https://github.com/qiuqiangkong/audioset_tagging_cnn |

---

## Libraries & Frameworks

| Library | Version (approx.) | Purpose |
|---|---|---|
| PyTorch | ≥ 2.0 | Model training, DataLoader, autograd |
| torchaudio | ≥ 2.0 | (dependency of audio processing) |
| librosa | ≥ 0.10 | Mel spectrogram computation, audio resampling |
| soundfile | ≥ 0.12 | Audio file reading |
| scikit-learn | ≥ 1.3 | Stratified train/val split, ROC-AUC metric |
| numpy | ≥ 1.24 | Numerical operations |
| pandas | ≥ 2.0 | CSV loading and metadata manipulation |
| matplotlib | ≥ 3.7 | Error analysis visualisations |
| jupyter | ≥ 1.0 | Notebook environment |

---

## AI Assistance Attribution

- Portions of this project including preprocessing logic, comments, helper functions, documentation wording, and structural recommendations were developed with assistance from:
  - Anthropic’s Claude
  - OpenAI’s ChatGPT
  - Google’s Gemini

- **Files containing AI-assisted contributions:**
  - `README.md` (content structure)
  - `SETUP.md`
  - `ATTRIBUTION.md`
  - `dataprocessing4.ipynb`
  - `oggtowav.ipynb`

- All AI-assisted code was reviewed, tested, and modified to ensure correctness and academic integrity