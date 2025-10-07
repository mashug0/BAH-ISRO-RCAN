# 🛰️ RADIAN: Noise-Aware Dual Image Super-Resolution for Satellite Imagery  
**Bharatiya Antariksh Hackathon 2025 – ISRO (Problem Statement 12)**  
**Team:** Mangal Mandli · *IIT Guwahati*  
**Members:** Saksham Gupta · Yashvi Mehta · Govind Akash Sanghani · Nalin Goel  

---

## 🚀 Overview

**RADIAN (Robust Alignment and Dual-image Attention Network)** is a deep learning framework for **dual-image super-resolution (SR)** of optical satellite imagery.  
The model leverages **two adjacent low-resolution (LR)** frames to reconstruct a **high-resolution (HR)** image with improved spatial fidelity and noise robustness.

This project was developed as part of the **Bharatiya Antariksh Hackathon 2025 (ISRO)** under *Problem Statement 12 – Dual Image Super-Resolution and Blind Evaluation.*

---

## 🧩 Core Architecture

RADIAN integrates two complementary modules for enhanced reconstruction:

### 1️⃣ Noise-Aware TDAN (Temporal Deformable Alignment Network)
- Aligns two degraded LR frames using **deformable convolutions**.  
- A **Noise Estimation Module** predicts pixel-wise noise maps that guide alignment.  
- The **Alignment Refinement Unit** ensures smooth spatial correspondence even under orbital jitter.

### 2️⃣ Lite RCAN (Residual Channel Attention Network)
- A lightweight version of RCAN optimized for memory and inference efficiency.  
- Employs **Channel Attention Blocks** and **Global Residual Fusion** to enhance textures and fine details.  
- Works jointly with TDAN to reconstruct HR images that balance fidelity and perceptual sharpness.

---

## ⚙️ Training & Evaluation Pipeline

1. **Synthetic Dataset Creation (`Dataset/`)**  
   - High-res images are degraded using a **physics-based model** (Gaussian blur, MTF noise, sub-pixel shifts, and random downsampling).  
   - The degradation process follows *Zhang et al., ICCV 2021* for realistic satellite conditions.

2. **Dual-Input SR Model (`Model/`)**  
   - Two LR frames → deformable alignment (Noise-Aware TDAN) → feature fusion → Lite RCAN reconstruction.  
   - **Loss Framework:**
     - Alignment Loss – temporal consistency  
     - Reconstruction Loss – pixel fidelity  
     - Noise Modeling Loss – aligns with estimated noise map  
     - Edge Preservation Loss – structural sharpness  
     - Perceptual Loss – enhances visual realism (LPIPS)

3. **Blind Image Quality Assessment (`BIQA/`)**  
   - A no-reference IQA model (BIECON + Gradient Sharpness features).  
   - Predicts perceptual quality without ground-truth HR references.  
   - Trained on synthetically degraded data for robustness.

---

## 🧠 Folder Structure
<pre>
BAH-ISRO-RCAN/
├── BIQA/                          # Blind Image Quality Assessment (No-reference evaluator)
│   ├── Degrador.py
│   ├── ExtractMos.py
│   ├── ExtractSSIM.py
│   ├── Train.py / Test.py
│   ├── model/
│   │   ├── BIECON.py
│   │   ├── BlindEvalModel.py
│   │   ├── GradientSharpness.py
│   │   └── TrainModel.py
│   └── mos.csv
│
├── Dataset/                       # Dataset creation and degradation scripts
│   ├── CreateDataset.py
│   ├── Degrador.py
│   └── Extractor.py
│
├── Model/                         # Core Dual-Image Super-Resolution Model
│   ├── NoiseAwareTDAN.py
│   ├── liteRCAN.py
│   ├── NoiseEstimator.py
│   ├── MultiComponentLoss.py
│   ├── DeformConv2d.py
│   └── Allignment.py
│
├── DatasetLoader.py               # Dataset loader utilities
├── MODEL_NOTEBOOK.ipynb           # 🔹 Main notebook to run Super-Resolution
└── PS12_Mangal Mandli .pptx       # Presentation slides (Hackathon submission)
</pre>


BIQA/model is not relevant to the Model Used. it will be updated soon.
best_model.pth, some checkpoint_epoch_num.pth fies are added to branch test-model


---

## 🧮 How to Run

### 🔹 For Super-Resolution:
Simply open and run the following notebook:

```bash
MODEL_NOTEBOOK.ipynb
```
This notebook covers:
Data loading

Model initialization (Noise-Aware TDAN + Lite RCAN)

Training & inference

Visualization of LR vs SR vs Ground Truth

No command-line execution is required — the full workflow runs inside the notebook.
Qualitative Results:

Sharper edges in roads, vegetation, and water boundaries.

Fine micro-textures recovered under noise and blur.

Strong alignment and perceptual consistency compared to single-frame SR baselines.

🧰 Tools & Frameworks

Deep Learning: PyTorch

Image Processing: OpenCV, PIL

Visualization: Matplotlib, tqdm

Evaluation: PSNR, SSIM, LPIPS, BRISQUE, NIQE

Optimization: Adam, EMA updates

Inference Acceleration: TensorRT supported


