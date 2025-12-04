# Reproducing Point Beyond Class (PBC) and Extending with Transformer Backbones

Authors:
- Sundeep Muthukrishnan Kumaraswamy
- Sai Snigdha Nadella

This repository reproduces the Point Beyond Class (PBC) benchmark from MICCAI 2022 on RSNA and VinDR-CXR, and extends it with a transformer backbone comparison (ResNet-50, ViT-Base, Swin-Tiny).

---

## 1. Background & Motivation

Bounding-box annotation is expensive, slow, and hard to scale.  
PBC reduces annotation cost by using ONE POINT per lesion instead of a bounding box.

Key Ideas:
- Point → Box DETR
- Multi-Point Consistency (MP)
- Symmetric Consistency (SC)
- Pseudo labels for student model

Project Goals:
1. Reproduce PBC results  
2. Compare Swin, ViT, ResNet backbones  
3. Identify best backbone for chest X-ray localization  

---

## 2. Repository Structure

Point-Beyond-Class/
├── data/
│   ├── RSNA/
│   ├── cxr/
├── datasets/
├── models/
│   ├── detr.py
│   ├── detr_swin.py
│   ├── detr_vit.py
├── student/
├── outfiles/logs/
├── pyScripts/
├── main.py
├── main_vit.py
├── main_swin.py
└── pbc_paper_env.yml

---

## 3. Dataset Preparation

RSNA Conversion:
cd data/RSNA
python rowimg2jpg.py
python row_csv2tgt_csv.py
python csv2coco_rsna.py

Output:
data/RSNA/cocoAnn/*p/*.json

VinDR-CXR Conversion:
cd data/cxr
python selectDetImgs.py
python generateCsvTxt.py
python csv2coco.py

Output:
data/cxr/ClsAll8_cocoAnnWBF/*p/*.json

---

## 4. PBC Pipeline

Stage 1 — Train Teacher (Point → Box)
Variants:
- Baseline
- MP
- SC
- SC-Pretrain + MP

Stage 2 — Generate Pseudo Boxes:
python main.py --eval --resume checkpoint.pth --save_csv pseudo.csv --generate_pseudo_bbox

Convert pseudo labels:
cd data/RSNA && python eval2train_RSNA.py
cd ../cxr && python eval2train_CXR8.py

Stage 3 — Student Model (MMDetection):
python3 tools/train.py configs/faster/faster_xxx.py --seed 42 --deterministic

---

## 5. Training Commands

Baseline Teacher (RSNA, 20%):
python main.py --epochs 111 --lr_backbone 1e-5 --lr 1e-4 --dataset_file rsna --coco_path data/RSNA/cocoAnn/20p --batch_size 16 --num_workers 16 --data_augment --position_embedding v4 --output_dir outfiles/models/RSNA/exp_baseline

Multi-Point Consistency:
--sample_points_num 2
--cons_loss
--cons_loss_coef 100

Symmetric Consistency:
--sample_points_num 1
--train_with_unlabel_imgs
--unlabel_cons_loss_coef 50
--partial 20

SC-Pretrain + MP:
--load_from checkpoint0110.pth

Backbone-specific:
python main_swin.py ...
python main_vit.py ...

---

## 6. Backbone Results (VinDR-CXR, 50%)

ResNet-50: 0.2930  
ViT-Base: 0.1982  
Swin-Tiny: 0.3069

Conclusion:
- Swin-Tiny is the best backbone  
- ViT lacks multi-scale representation  
- Swin is more stable under MP + SC  

---

## 7. Visualization

cd pyScripts
python drawLogCXR8.py
python drawLogRSNA.py

---

## 8. Future Work

- CXR-specific Transformer  
- CNN + Transformer hybrid  
- Better pseudo labels using uncertainty  
- Multi-dataset evaluation (NIH, CheXpert, MIMIC-CXR)  
- Explainability: Grad-CAM, attention rollout  
- Radiologist-in-the-loop learning  

---

## 9. Citation

@inproceedings{ji2022point,
  title={Point Beyond Class: A Benchmark for Weakly Semi-supervised Abnormality Localization in Chest X-Rays},
  author={Ji, Haoqin and Liu, Haozhe and Li, Yuexiang and Xie, Jinheng and He, Nanjun and Huang, Yawen and Wei, Dong and Chen, Xinrong and Shen, Linlin and Zheng, Yefeng},
  booktitle={MICCAI},
  year={2022}
}

@article{ji2022benchmark,
  title={A Benchmark for Weakly Semi-Supervised Abnormality Localization in Chest X-Rays},
  author={Ji, Haoqin and Liu, Haozhe and Li, Yuexiang and Xie, Jinheng and He, Nanjun and Huang, Yawen and Wei, Dong and Chen, Xinrong and Shen, Linlin and Zheng, Yefeng},
  journal={arXiv preprint arXiv:2209.01988},
  year={2022}
}
