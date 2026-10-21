#  Clinically Calibrated Chest X-Ray Screening Engine

An enterprise-grade deep learning system for chest radiograph pneumonia triage, featuring patient-grouped stratification, native PyTorch Grad-CAM explainability, and Platt probability calibration.

##  Features
- **Patient-Level Data Hygiene:** Patient-grouped 80/10/10 split using `StratifiedGroupKFold` across 4,800+ patient IDs to eliminate data leakage.
- **Model Benchmark:** Benchmarked Baseline CNN, ResNet50, and DenseNet121 across frozen, partial, and full fine-tuning strategies.
- **Native Explainable AI (XAI):** Built-in Grad-CAM hook engine rendering real-time feature attribution heatmaps without external dependencies.
- **Calibrated Triage Threshold:** Post-hoc Platt Scaling tuned to operating threshold $\tau = 0.28$ (Sensitivity $\ge 95\%$) to minimize missed diagnoses.