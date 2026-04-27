# SETUP — Pantanal BirdCLEF 2026

Follow these steps in order to reproduce the environment and run the notebook.

---

## 1. Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.9+ |
| CUDA-capable GPU | Recommended (CPU works but is very slow) |
| CUDA toolkit | 11.8+ (if using GPU) |
| conda or pip | Either works; conda recommended for audio libs |

---

## 2. Clone the repository

```bash
git clone https://github.com/sophiasang13/sangcs372final.git
cd Finalproject
```
---

# 3. Install the AudioSet CNN14 repository

The backbone model lives in a separate repo 

```bash
git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git
```

Place (or symlink) the cloned folder at the path expected by the notebook:
```
<project-root>/audioset_tagging_cnn/
```

The notebook adds both `audioset_tagging_cnn/` and `audioset_tagging_cnn/pytorch/` to `sys.path` automatically.

---

## 4. Download competition data

Download the following files from the [BirdCLEF 2026 Kaggle competition page](https://www.kaggle.com/competitions/birdclef-2026) and place them in the project root:

```
train.csv
train_soundscapes_labels.csv
taxonomy.csv
```

---

## 5. Download the pretrained CNN14 checkpoint

Download `Cnn14_mAP=0.431.pth` from the [PANNs model zoo](https://zenodo.org/record/3987831) and place it in the project root:

```
Cnn14_mAP=0.431.pth
```

---

## 6. Prepare audio files

```bash
jupyter notebook oggtowav.ipynb
```
```python
input_dir = "/home/users/ss1482/Final project/birdtrainold"
output_dir = "/home/users/ss1482/Final project/birdtrain_wav"
```
Change input_dir to where your ogg files are hosted and output directory to where you would like to store the wav files

Run all cells top-to-bottom. Training checkpoints are saved automatically to the project root.

## 8. Launch the main notebook 

```bash
jupyter notebook dataprocessing4.ipynb
```

Run all cells top-to-bottom. Training checkpoints are saved automatically to the project root.

## 9. Launch the webapp

```bash
jupyter notebook Inference.ipynb
```

Run all cells top-to-bottom. Training checkpoints are saved automatically to the project root.

---


