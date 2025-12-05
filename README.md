# Reproducing the Point Beyond Class (PBC) Teacher Model  
## RSNA & VinDR-CXR — Baseline, Multi-Point, and Symmetric Consistency  

**Authors:** Sundeep Muthukrishnan Kumaraswamy (B01105889), Sai Snigdha Nadella (B01123360)

---

## 1. Overview

This repository reproduces the **Teacher component** of the MICCAI 2022 paper:  
**"Point Beyond Class: A Benchmark for Weakly Semi-Supervised Abnormality Localization in Chest X-Rays."** [web:1][web:2]

We reproduced and extended the **PBC Teacher Model**, including:
- **Baseline (Point → Box)**
- **Multi-Point Consistency (MP)**
- **Symmetric Consistency (SC)**
- **Final Teacher (SC + MP Combined)**

All experiments were performed on **RSNA Pneumonia** and **VinDR-CXR (CXR8)** datasets.  
We further extended PBC with **Transformer-based backbones (ViT-Base and Swin-Tiny)** to analyze backbone sensitivity in semi-supervised lesion localization. [web:3]

---

## 2. Environment Setup

Two environments were used:

### 2.1 Original PBC (ResNet Teacher)
Used for standard PBC teacher reproduction.

