````markdown
# Reproducing the Point Beyond Class (PBC) Teacher Model  
### RSNA & VinDR-CXR — Baseline, Multi-Point, and Symmetric Consistency  
**Authors:** Sundeep Muthukrishnan Kumaraswamy (B01105889), Sai Snigdha Nadella (B01123360)

---

## 1. Overview

This repository reproduces the **Teacher component** of the MICCAI 2022 paper:  
**“Point Beyond Class: A Benchmark for Weakly Semi-Supervised Abnormality Localization in Chest X-Rays.”**

We reproduced and extended the **PBC Teacher Model**, including:
- **Baseline (Point → Box)**
- **Multi-Point Consistency (MP)**
- **Symmetric Consistency (SC)**
- **Final Teacher (SC + MP Combined)**

All experiments were performed on **RSNA Pneumonia** and **VinDR-CXR (CXR8)** datasets.  
We further extended PBC with **Transformer-based backbones (ViT-Base and Swin-Tiny)** to analyze backbone sensitivity in semi-supervised lesion localization.

---

## 2. Environment Setup

Two environments were used:

### 2.1 Original PBC (ResNet Teacher)
Used for standard PBC teacher reproduction.

```bash
conda env create -f pbc_paper_env.yml
conda activate pbc_paper
````

### 2.2 Transformer Backbones (ViT + Swin)

Required for:

* `main_swin.py`
* `main_vit.py`

```bash
conda create -n pbc_swin_clean python=3.8
conda activate pbc_swin_clean
# install torch, torchvision, timm, mmcv, mmdet, etc.
```

> **Note:** Always activate `pbc_swin_clean` before running Swin or ViT experiments to avoid layer normalization or timm dependency errors.

---

## 3. Dataset Preparation

We used **RSNA Pneumonia Detection** and **VinDR-CXR (CXR8)** datasets.
Each dataset must be converted to **COCO format**.

### 3.1 RSNA → COCO

```bash
cd data/RSNA
python rowimg2jpg.py
python row_csv2tgt_csv.py
python csv2coco_rsna.py
```

Generated annotations:

```
data/RSNA/cocoAnn/*p/*.json
```

### 3.2 VinDR-CXR → COCO

```bash
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

Each backbone uses a corresponding training file.

| Backbone  | Script         |
| --------- | -------------- |
| ResNet-50 | `main.py`      |
| ViT-Base  | `main_vit.py`  |
| Swin-Tiny | `main_swin.py` |

---

### 4.1 Baseline (Point → Box)

```bash
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

---

### 4.2 Multi-Point Consistency (MP)

```bash
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

---

### 4.3 Symmetric Consistency (SC)

```bash
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

---

### 4.4 Final Teacher (SC + MP Combined)

```bash
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

### 5.1 RSNA (AP50)

| Model       | Paper      | Reproduced   |
| ----------- | ---------- | ------------ |
| Baseline    | 2.1 → 25.1 | 2.46 → 24.59 |
| Multi-Point | 3.4 → 29.1 | matched      |
| Symmetric   | 4.1 → 32.4 | matched      |
| SC + MP     | ~33        | matched      |

---

### 5.2 VinDR-CXR (AP50)

| Model       | Paper | Reproduced |
| ----------- | ----- | ---------- |
| Baseline    | 12.3  | 12.26      |
| Multi-Point | 21.9  | 21.79      |
| Symmetric   | 11.8  | 15.55      |
| SC + MP     | 28.5  | 29.37      |

---

## 6. Transformer Backbone Extension

### ViT-Base (`main_vit.py`)

* Works fine for baseline
* Poor under consistency losses (no hierarchical context)
* Limited for multi-scale X-ray features

### Swin-Tiny (`main_swin.py`)

* Best-performing backbone
* Robust to sparse supervision
* Superior AP50 under SC + MP

> **Run using:**
>
> ```bash
> conda activate pbc_swin_clean
> ```

---

## 7. Logs & Visualization

Logs:

```
outfiles/logs/RSNA/
outfiles/logs/CXR8/
```

Visualize training:

```bash
cd pyScripts
python drawLogRSNA.py
python drawLogCXR8.py
```

---

## 8. Repository Structure

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

---

## 9. References

**Paper:**

```
@inproceedings{ji2022point,
  title={Point Beyond Class: A Benchmark for Weakly Semi-supervised Abnormality Localization in Chest X-Rays},
  author={Ji, Haoqin and Liu, Haozhe and Li, Yuexiang and Xie, Jinheng and He, Nanjun and Huang, Yawen and Wei, Dong and Chen, Xinrong and Shen, Linlin and Zheng, Yefeng},
  booktitle={MICCAI},
  year={2022}
}
```

---

## 10. Acknowledgment

This project reproduces and extends the **Point Beyond Class (PBC)** teacher model.
We thank the original authors and related works such as **UP-DETR** and **Point-DETR** for their codebase foundations.

```
```
