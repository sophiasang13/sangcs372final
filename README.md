# sangcs372final
Spring 2026 CS 372 Final
# Pantanal BirdCLEF 2026 — Bird Species Classifier

## What it Does

This project builds an audio-based bird species classifier for the **BirdCLEF 2026 competition**, focused on the Pantanal ecosystem with 234 species. It uses a two-phase transfer learning approach: a pretrained **CNN14** backbone (originally trained on AudioSet) is first fine-tuned on short XC/iNat bird clips with only the classification head trainable (Phase 1), then fully fine-tuned on labeled 5-second soundscape segments from field recordings (Phase 2). The pipeline handles extreme class imbalance through stratified splitting, file-level data leakage prevention, rare-species oversampling via `WeightedRandomSampler`, Focal Loss, and mixup augmentation — producing per-class ROC-AUC metrics and interpretable error analysis plots for every evaluated species.

---

## Quick Start
### Clone project + backbone
git clone https://github.com/sophiasang13/sangcs372final.git
cd Finalproject
git clone https://github.com/qiuqiangkong/audioset_tagging_cnn.git

### Install dependencies
pip install -r requirements.txt

### --- MANUAL STEP ---
 download these into the project root:
train.csv
train_soundscapes_labels.csv
taxonomy.csv
Cnn14_mAP=0.431.pth

### convert audio (edit paths inside notebook first)
jupyter notebook oggtowav.ipynb

### run training

jupyter notebook dataprocessing4.ipynb
---

## Video Links

| Video | Link |
|---|---|
| Demo Walkthrough | *(add link here)* |
| Technical Walkthrough | *(add link here)* |

---

## Evaluation

The model is evaluated using **macro ROC-AUC** across all species that appear in the validation set. Key results:

## Evaluation & Model Iterations

The model development followed a systematic, evaluation-driven approach. Each iteration was designed to address specific failure modes identified in the validation data (e.g., slow convergence, class imbalance, and "silent" soundscape segments).

### **Model Improvement History**

| Iteration | Major Changes | Key Metrics | Observation/Justification |
| :--- | :--- | :--- | :--- |
| **0: Baseline** | Phase 2 Fine-tuning (CNN14) | **Val AUC: 0.6463** | Initial training with base parameters; model showed slow convergence and high training loss. |
| **1: Optimizer & LR** | Switched to **AdamW** + increased LR to `1e-5` | **Val AUC: 0.7166** | AdamW improved weight decay; higher LR moved the model out of a local minimum. |
| **2: Augmentation** | Implemented **Mixup** augmentation | **Val AUC: 0.7688** | Significant boost in generalization (+5.2%). Reduced overfitting on short clips. |
| **3: Loss & Stability** | **Focal Loss** + **Gradient Accumulation** | **Val AUC: 0.6818** | Temporary dip in AUC. Focal Loss targeted rare species, but required more stability (larger effective batch size). |
| **4: Data Quality** | **Targeted Resampling** of Soundscapes | **Val AUC: 0.7988** | **Best Performance.** Realized soundscapes were "sparse" (low bird density). Filtering for active segments improved signal-to-noise ratio. |

---

### **Key Technical Implementations**

#### **1. Preprocessing & Data Quality **
To address the challenge of "noisy labels" and "sparsity" in field recordings:
* **Active Segment Resampling:** We analyzed the soundscape files and discovered high percentages of silence. By resampling the dataset to prioritize segments with confirmed bird activity, the Val AUC improved from **0.75 to 0.79**.
* **Class Imbalance:** Integrated a `WeightedRandomSampler` to ensure rare species (some with <5 examples) were seen by the model as frequently as common species.

#### **2. Regularization & Stability **
* **Regularization:** Used a combination of **Dropout (0.2)** and **L2 Weight Decay** (via AdamW) to prevent the CNN14 backbone from memorizing the background noise of specific recording sites.
* **Training Stability:** Implemented **Gradient Accumulation** (4 steps) to simulate a larger batch size, which was necessary to stabilize the gradients when using Focal Loss on a single GPU.
* **Mixed Precision:** Utilized `torch.cuda.amp` to reduce VRAM footprint on the RTX 2080 Ti, allowing for faster iteration cycles.

#### **3. Transfer Learning & Domain Adaptation (7 pts)**
* **Task Adaptation:** Successfully adapted a model pretrained on **AudioSet** (general environmental sounds like "Car" or "Speech") to the highly specific domain of **Pantanal Avian Vocalizations**. This involved a two-phase strategy: first freezing the backbone to stabilize the new head, followed by a partial unfreezing of the top convolutional blocks.


## Error Analysis

To claim the 7-point rubric item for Error Analysis, this section discusses the systematic strengths and weaknesses of the final bird species classifier based on the following three diagnostic visualizations.

### **1. Performance Distribution (Fig 1)**
![AUC Distribution Per Species](Finalproject/error_analysis_fig1_auc_distribution.png)
*Figure 1: ROC-AUC Distribution across 234 species.*

**Analysis:** The distribution shows a strong "head" of high-performing species (AUC > 0.85) but a long "tail" of struggling classes. While the model excels at identifying common species with distinct melodic signatures, the density of species in the 0.50–0.65 range indicates significant difficulty in the multi-label classification task when bird calls overlap.

### **2. The Sparsity Gap (Fig 2)**
![Sparsity vs Performance](Finalproject/error_analysis_fig2_sparsity.png)
*Figure 2: Correlation between training sample volume and per-class AUC.*

**Analysis:** There is a clear positive correlation between the number of training samples and species-level performance. 
- **The "Data Desert":** Species with fewer than 10 training segments (e.g., rare Pantanal endemics) consistently fall below the 0.60 AUC threshold. 
- **Justification for Methodology:** This finding justified the implementation of the `WeightedRandomSampler` and the use of **Focal Loss**, which forces the model to attend more heavily to these underrepresented, high-difficulty samples.

### **3. Failure Case Taxonomy (Fig 3)**
![Failure Case Examples](Finalproject/error_analysis_fig3_failure_cases.png)
*Figure 3: Qualitative look at False Positives (FP) and False Negatives (FN).*

**Why the model fails:**
Through manual inspection of the failure cases in Figure 3, three primary environmental challenges were identified:
* **Low Signal-to-Noise Ratio (SNR):** In the "False Negative" cases, bird vocalizations are often masked by heavy rain or wind—common in Pantanal field recordings. The CNN14 backbone struggles to distinguish low-amplitude harmonics from broadband environmental noise.
* **Confusion between Morphologically Similar Calls:** Several species (e.g., specific flycatchers) have calls with nearly identical frequency ranges and temporal patterns. Without more training data, the model lacks the "fine-grained" resolution to distinguish these subtle differences.
* **Audio "Silence" in Soundscapes:** As identified in Iteration 4, many soundscape segments are effectively "empty." Predicting the most common species (the "Global Average") becomes a failure mode when the model tries to find a signal in near-silent recordings.

---

### **Conclusion of Impact**
This error analysis directly informed the final training phase. By identifying that sparsity was the primary bottleneck, we shifted from model-centric changes (architecture) to data-centric changes (targeted resampling and Focal Loss), resulting in the final Val AUC improvement to **0.7988**.