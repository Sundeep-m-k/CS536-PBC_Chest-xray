# ========================================================================
# README.sh — Point Beyond Class (PBC) Reproduction + Backbone Extension
# ========================================================================
# Authors:
#   - Sundeep Muthukrishnan Kumaraswamy
#   - Sai Snigdha Nadella
#
# Description:
#   This bash-formatted README contains the complete documentation for
#   reproducing the Point Beyond Class (PBC) benchmark and extending it
#   using transformer backbones (ResNet-50, ViT-Base, Swin-Tiny).
#
#   EVERYTHING is placed inside this bash file as comments + runnable code.
# ========================================================================


# ========================================================================
# 1. BACKGROUND & MOTIVATION
# ========================================================================
# PBC solves the problem of expensive chest X-ray annotations by replacing
# bounding boxes with a *single point annotation per lesion*.
#
# Key Ideas:
#   - Modify DETR for Point → Box learning
#   - Add Multi-Point Consistency (MP)
#   - Add Symmetric Consistency (SC)
#   - Use pseudo-labels to train a strong student model
#
# Project Goals:
#   1. Reproduce PBC teacher model on RSNA & VinDR
#   2. Extend with transformer backbones (ViT, Swin, ResNet)
#   3. Identify best backbone for weakly-supervised localization
# ========================================================================


# ========================================================================
# 2. REPOSITORY LAYOUT
# ========================================================================
# Point-Beyond-Class/
# ├── data/
# │   ├── RSNA/
# │   ├── cxr/
# ├── datasets/
# ├── models/
# │   ├── detr.py
# │   ├── detr_vit.py
# │   ├── detr_swin.py
# ├── student/
# ├── outfiles/logs/
# ├── pyScripts/
# │   ├── drawLogCXR8.py
# │   ├── drawLogRSNA.py
# ├── main.py
# ├── main_vit.py
# ├── main_swin.py
# ├── start_RSNA.sh
# ├── start_CXR8.sh
# └── pbc_paper_env.yml
# ========================================================================


# ========================================================================
# 3. DATASET PREPARATION
# ========================================================================

# --------------------- RSNA CONVERSION ---------------------
echo "Converting RSNA dataset..."
cd data/RSNA || exit
python rowimg2jpg.py
python row_csv2tgt_csv.py
python csv2coco_rsna.py
cd ../../

# Output stored in:
#   data/RSNA/cocoAnn/*p/*.json


# ---------------------- CXR8 / VinDR -----------------------
echo "Converting CXR8 / VinDR dataset..."
cd data/cxr || exit
python selectDetImgs.py
python generateCsvTxt.py
python csv2coco.py
cd ../../

# Output stored in:
#   data/cxr/ClsAll8_cocoAnnWBF/*p/*.json
# ========================================================================


# ========================================================================
# 4. PBC TRAINING PIPELINE
# ========================================================================

# STAGE 1 — TEACHER MODEL (POINT → BOX)
#   Variants:
#     - Baseline
#     - Multi-Point Consistency (MP)
#     - Symmetric Consistency (SC)
#     - SC Pretrain + MP

# STAGE 2 — PSEUDO-LABEL GENERATION

# STAGE 3 — STUDENT MODEL (MMDETECTION)
# ========================================================================


# ========================================================================
# 5. TRAINING COMMANDS
# ========================================================================

# ------------------ BASELINE TEACHER EXAMPLE ------------------
echo "Running Baseline Teacher Model (RSNA, 20%)..."

partial=20
python main.py \
  --epochs 111 \
  --lr_backbone 1e-5 \
  --lr 1e-4 \
  --dataset_file rsna \
  --coco_path data/RSNA/cocoAnn/${partial}p \
  --batch_size 16 \
  --num_workers 16 \
  --data_augment \
  --position_embedding v4 \
  --output_dir outfiles/models/RSNA/exp_baseline


# ------------------ MULTI-POINT CONSISTENCY -------------------
# Add these flags:
#   --sample_points_num 2
#   --cons_loss
#   --cons_loss_coef 100


# ------------------ SYMMETRIC CONSISTENCY ---------------------
# Add these flags:
#   --sample_points_num 1
#   --train_with_unlabel_imgs
#   --unlabel_cons_loss_coef 50
#   --partial ${partial}


# ------------------ SC PRETRAINING + MP -----------------------
# Add flag:
#   --load_from checkpoint0110.pth


# ------------------ SWIN & VIT TRAINING -----------------------
# Swin Transformer:
#   python main_swin.py ...
#
# ViT:
#   python main_vit.py ...
# ========================================================================


# ========================================================================
# 6. GENERATING PSEUDO LABELS
# ========================================================================
echo "Generating pseudo-labels..."

python main.py \
  --eval \
  --resume checkpoint.pth \
  --save_csv pseudo.csv \
  --generate_pseudo_bbox

cd data/RSNA && python eval2train_RSNA.py
cd ../cxr && python eval2train_CXR8.py
cd ../../
# ========================================================================


# ========================================================================
# 7. STUDENT MODEL TRAINING (MMDETECTION)
# ========================================================================
# Example:
# python3 tools/train.py configs/faster/faster_xxx.py \
#   --seed 42 \
#   --deterministic
# ========================================================================


# ========================================================================
# 8. BACKBONE COMPARISON RESULTS
# ========================================================================
# Dataset: VinDR-CXR, 50% training data
#
# BACKBONE PERFORMANCE (AP50):
#   ResNet-50     = 0.2930
#   ViT-Base      = 0.1982
#   Swin-Tiny     = 0.3069   <-- BEST
#
# KEY INSIGHTS:
#   - ViT fails due to lack of multi-scale features
#   - Swin has hierarchical windows + strong local attention
#   - Swin is most stable under MP + SC
#   - Swin is recommended backbone for medical images
# ========================================================================


# ========================================================================
# 9. LOG VISUALIZATION
# ========================================================================
cd pyScripts || exit

echo "Plotting logs..."
python drawLogCXR8.py
python drawLogRSNA.py

cd ../ || exit
# ========================================================================


# ========================================================================
# 10. FUTURE WORK
# ========================================================================
# - Build a custom medical transformer
# - Explore CNN + Transformer hybrid models
# - Improve pseudo labels using uncertainty estimation
# - Evaluate on NIH, CheXpert, MIMIC-CXR
# - Add explainability (Grad-CAM, attention visualization)
# - Develop radiologist-in-the-loop active learning
# ========================================================================


# ========================================================================
# 11. CITATIONS
# ========================================================================
# @inproceedings{ji2022point,
#   title={Point Beyond Class: A Benchmark for Weakly Semi-supervised Abnormality Localization in Chest X-Rays},
#   author={Ji, Haoqin and Liu, Haozhe and Li, Yuexiang and Xie, Jinheng and He, Nanjun and Huang, Yawen and Wei, Dong and Chen, Xinrong and Shen, Linlin and Zheng, Yefeng},
#   booktitle={MICCAI},
#   pages={249--260},
#   year={2022},
#   organization={Springer}
# }
#
# @article{ji2022benchmark,
#   title={A Benchmark for Weakly Semi-Supervised Abnormality Localization in Chest X-Rays},
#   author={Ji, Haoqin and Liu, Haozhe and Li, Yuexiang and Xie, Jinheng and He, Nanjun and Huang, Yawen and Wei, Dong and Chen, Xinrong and Shen, Linlin and Zheng, Yefeng},
#   journal={arXiv preprint arXiv:2209.01988},
#   year={2022}
# }
# ========================================================================

# END OF README.sh
