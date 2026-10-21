# 🩺 Clinical Pneumonia AI Screening & Triage Pipeline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An enterprise-grade medical computer vision system designed to assist radiologists by performing automated triage, epistemic uncertainty estimation, spatial interpretability (Grad-CAM), and input quality verification on posteroanterior (PA) chest radiographs.

---

## 🏛 System Architecture

```text
Chest_Xray_Pneumonia_AI/
├── src/
│   ├── dataset.py       # Patient-grouped data split & Albumentations pipeline
│   ├── models.py        # ResNet50 backbone initialization & fine-tuning logic
│   ├── gradcam.py       # PyTorch hook-based Grad-CAM feature attributions
│   ├── guardrails.py    # Laplacian variance blur & color-profile validation
│   ├── inference.py     # Monte Carlo Dropout (N=15) uncertainty estimator
│   └── report.py        # PDF diagnostic report compiler (ReportLab)
├── app.py               # Interactive Streamlit clinical web application
├── resnet50_pneumonia.pth # Trained model weights
└── requirements.txt     # Environment dependencies