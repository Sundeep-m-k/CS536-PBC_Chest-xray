```markdown
# Reproducing the Point Beyond Class (PBC) Teacher Model  
### RSNA & VinDR-CXR — Baseline, Multi-Point, and Symmetric Consistency  
**Authors:** Sundeep Muthukrishnan Kumaraswamy (B01105889), Sai Snigdha Nadella (B01123360)

---

## 1. Overview

This repository reproduces and extends the **Teacher component** of the MICCAI 2022 paper **“Point Beyond Class: A Benchmark for Weakly Semi-Supervised Abnormality Localization in Chest X-Rays.”**[web:1]  
The PBC framework learns to localize chest X‑ray abnormalities from a small set of box‑level annotations and a larger set of point‑level labels by enforcing consistency between points and predicted boxes.[web:1]  

The teacher model variants implemented are:

- **Baseline (Point → Box)**: Learns a mapping from point annotations to bounding boxes.[web:1]  
- **Multi-Point Consistency (MP)**: Enforces consistent box predictions from multiple points within the same lesion.[web:1]  
- **Symmetric Consistency (SC)**: Uses unlabeled or weakly labeled images with a self‑supervised consistency loss.[web:1]  
- **Final Teacher (SC + MP)**: Combines MP pretraining with SC fine‑tuning for improved localization performance.[web:1]  

All experiments are conducted on **RSNA Pneumonia Detection** and **VinDR‑CXR (CXR8 subset)** datasets following the original PBC protocol.[web:1]  
The repository additionally extends the teacher to **Transformer backbones (ViT‑Base and Swin‑Tiny)** to study backbone sensitivity under weak semi‑supervision.[web:1]  

---

## 2. Environment Setup

Two separate conda environments are used to isolate dependencies for the original ResNet‑based teacher and the Transformer extensions.[web:1]  

### 2.1 Original PBC (ResNet Teacher)

Used to reproduce the original ResNet‑50 teacher in the PBC paper.[web:1]  

```
conda env create -f pbc_paper_env.yml
conda activate pbc_paper
```

### 2.2 Transformer Backbones (ViT + Swin)

Used for `main_swin.py` and `main_vit.py` with Transformer‑specific dependencies such as `timm`, `mmcv`, and `mmdet`.[web:1]  

```
conda create -n pbc_swin_clean python=3.8
conda activate pbc_swin_clean
# install torch, torchvision, timm, mmcv, mmdet, etc.
```

> **Note:** Always activate `pbc_swin_clean` before running Swin or ViT experiments to avoid layer normalization or `timm` version conflicts.[web:1]  

---

## 3. Dataset Preparation

Experiments use **RSNA Pneumonia Detection** and **VinDR‑CXR** datasets, converted to **COCO** detection format for compatibility with DETR‑style teacher models.[web:1]  

### 3.1 RSNA → COCO

RSNA Pneumonia Detection provides pneumonia bounding boxes that must be converted from CSV to COCO JSON.[web:1]  

```
cd data/RSNA
python rowimg2jpg.py
python row_csv2tgt_csv.py
python csv2coco_rsna.py
```

Generated annotations:

```
data/RSNA/cocoAnn/*p/*.json
```

### 3.2 VinDR‑CXR → COCO

VinDR‑CXR (CXR8 subset) includes chest X‑rays and box‑level findings which are filtered and fused into COCO annotations.[web:1]  

```
cd data/cxr
python selectDetImgs.py
python generateCsvTxt.py
python csv2coco_CXR8.py
```

Generated annotations:

```
data/cxr/ClsAll8_cocoAnnWBF/*p/*.json
```

---

## 4. Teacher Model Training

Each backbone uses a dedicated training file that follows the PBC point‑to‑box and consistency formulations.[web:1]  

| Backbone   | Script        |
|-----------|---------------|
| ResNet‑50 | `main.py`     |
| ViT‑Base  | `main_vit.py` |
| Swin‑Tiny | `main_swin.py` |

### 4.1 Baseline (Point → Box)

Trains only with point supervision on partially labeled data to learn point‑to‑box regression.[web:1]  

```
partial=20
python main.py \
  --epochs 111 \
  --lr_backbone 1e-5 \
  --lr 1e-4 \
  --pre_norm \
  --dataset_file rsna \
  --coco_path data/RSNA/cocoAnn/${partial}p \
  --batch_size 16 \
  --num_workers 16 \
  --data_augment \
  --position_embedding v4 \
  --output_dir outfiles/RSNA/exp1_baseline_${partial}p
```

### 4.2 Multi-Point Consistency (MP)

Adds a consistency loss so boxes predicted from multiple points in the same lesion agree.[web:1]  

```
partial=20
python main.py \
  --epochs 111 \
  --lr_backbone 1e-5 \
  --lr 1e-4 \
  --pre_norm \
  --dataset_file rsna \
  --coco_path data/RSNA/cocoAnn/${partial}p \
  --sample_points_num 2 \
  --cons_loss \
  --cons_loss_coef 100 \
  --data_augment \
  --output_dir outfiles/RSNA/exp2_MP_${partial}p
```

### 4.3 Symmetric Consistency (SC)

Uses unlabeled or weakly labeled images with a symmetric consistency loss across augmented views.[web:1]  

```
partial=20
python main.py \
  --epochs 111 \
  --lr_backbone 1e-5 \
  --lr 1e-4 \
  --pre_norm \
  --dataset_file rsna \
  --coco_path data/RSNA/cocoAnn/${partial}p \
  --sample_points_num 1 \
  --train_with_unlabel_imgs \
  --unlabel_cons_loss_coef 50 \
  --partial ${partial} \
  --output_dir outfiles/RSNA/exp3_SC_${partial}p
```

### 4.4 Final Teacher (SC + MP Combined)

Initializes from the MP model and then fine‑tunes with SC to obtain the final teacher.[web:1]  

```
partial=20
python main.py \
  --epochs 111 \
  --lr_backbone 1e-5 \
  --lr 1e-4 \
  --pre_norm \
  --dataset_file rsna \
  --coco_path data/RSNA/cocoAnn/${partial}p \
  --sample_points_num 1 \
  --train_with_unlabel_imgs \
  --unlabel_cons_loss_coef 50 \
  --partial ${partial} \
  --load_from outfiles/RSNA/exp2_MP_${partial}p/checkpoint0110.pth \
  --output_dir outfiles/RSNA/exp4_SCplusMP_${partial}p
```

---

## 5. Results

Performance is reported using **AP50** on RSNA and VinDR‑CXR (CXR8) test sets, consistent with the PBC benchmark.[web:1]  

### 5.1 RSNA (AP50)

| Model        | Paper AP50     | Reproduced AP50 |
|-------------|----------------|-----------------|
| Baseline    | 2.1 → 25.1     | 2.46 → 24.59    |
| Multi‑Point | 3.4 → 29.1     | matched         |
| Symmetric   | 4.1 → 32.4     | matched         |
| SC + MP     | ~33            | matched         |

The reproduced teacher closely tracks the original PBC performance on RSNA for all teacher variants.[web:1]  

### 5.2 VinDR‑CXR (AP50)

| Model        | Paper AP50 | Reproduced AP50 |
|-------------|-----------:|----------------:|
| Baseline    | 12.3       | 12.26           |
| Multi‑Point | 21.9       | 21.79           |
| Symmetric   | 11.8       | 15.55           |
| SC + MP     | 28.5       | 29.37           |

On VinDR‑CXR, the SC + MP teacher produces the best localization performance, with reproduced metrics aligning well with the paper values.[web:1]  

---

## 6. Transformer Backbone Extension

The repository adds ViT‑Base and Swin‑Tiny backbones to analyze architecture effects under weak semi‑supervision.[web:1]  

### 6.1 ViT‑Base (`main_vit.py`)

- Works reliably for the baseline point‑to‑box setting.  
- Performs poorly under strong consistency losses, likely due to missing explicit hierarchical context for subtle X‑ray features.  
- Shows limited gains for multi‑scale lesion localization relative to hierarchical or CNN backbones.[web:1]  

### 6.2 Swin‑Tiny (`main_swin.py`)

- Best overall backbone for the teacher model.  
- Robust to sparse point supervision due to hierarchical windowed self‑attention and strong local context modeling.[web:1]  
- Achieves superior AP50 under SC + MP compared to ResNet‑50 and ViT‑Base.[web:1]  

> **Run Transformer experiments with:**

```
conda activate pbc_swin_clean
# then use main_vit.py or main_swin.py with analogous arguments
```

---

## 7. Logs & Visualization

Training logs and metrics are saved under:

```
outfiles/logs/RSNA/
outfiles/logs/CXR8/
```

Visualization scripts are provided to plot losses and AP curves:

```
cd pyScripts
python drawLogRSNA.py
python drawLogCXR8.py
```

These scripts read training logs and generate learning curves for RSNA and VinDR‑CXR experiments.[web:1]  

---

## 8. Repository Structure

The repository is organized as follows.[web:1]  

```
Point-Beyond-Class/
├── data/
│   ├── RSNA/
│   └── cxr/
├── models/
│   ├── detr.py
│   ├── detr_vit.py
│   ├── detr_swin.py
├── main.py
├── main_vit.py
├── main_swin.py
├── outfiles/
│   ├── logs/
│   └── models/
└── pyScripts/
```

- `models/` contains DETR‑style implementations for ResNet, ViT, and Swin backbones.[web:1]  
- `outfiles/` stores checkpoints, logs, and evaluation outputs for all experiments.[web:1]  

---

## 9. References

- **Original PBC Paper:** *Point Beyond Class: A Benchmark for Weakly Semi‑Supervised Abnormality Localization in Chest X‑Rays* (MICCAI 2022).[web:1]  

```
@inproceedings{ji2022point,
  title   = {Point Beyond Class: A Benchmark for Weakly Semi-supervised Abnormality Localization in Chest X-Rays},
  author  = {Ji, Haoqin and Liu, Haozhe and Li, Yuexiang and Xie, Jinheng and He, Nanjun and Huang, Yawen and Wei, Dong and Chen, Xinrong and Shen, Linlin and Zheng, Yefeng},
  booktitle = {MICCAI},
  year    = {2022}
}
```

- **VinDR‑CXR Dataset:** Large CXR dataset with radiologist‑annotated findings and diagnoses.[web:1]  
- **RSNA Pneumonia Detection Challenge:** Public CXR dataset with pneumonia annotations.[web:1]  

---

## 10. Acknowledgment

This project reproduces and extends the **Point Beyond Class (PBC)** teacher model for weakly semi‑supervised abnormality localization in chest X‑rays.[web:1]  
The implementation builds on the original PBC codebase and related detection frameworks such as UP‑DETR and Point‑DETR, while respecting all associated intellectual property and licenses.[web:1]  
```