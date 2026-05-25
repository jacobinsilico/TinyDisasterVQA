# COCO-QA Supervised VQA and Text-Embedding Prototype Cosine Pipeline

This repository implements a deployable visual-question answering (VQA) pipeline tailored for edge devices (such as GAP9). It supports dual-head classification modes, training-only question vocab generation, mixed precision acceleration, and lightweight text-embedding classification based on cosine prototype mappings.

---

## Workspace Directory Structure

- `src/data/dataset.py`: Fully whitelisted 40 object and 10 color category dataset loader.
- `src/models/vqa_models.py`: Dual-mode visual-question model (`classifier` and `prototype` head modes) with separate classifier heads and logit scaling.
- `scripts/build_answer_prototypes.py`: Offline text-embedding projector with deterministically seeded hash fallback vector capabilities.
- `scripts/train_question_vqa.py`: Colab-ready, mixed-precision enabled supervised training baseline runner.
- `scripts/smoke_test_prototype_model.py`: Dimensional shape, L2-normalization, and loss verification checks.

---

## Google Colab Sanity Command Guide

Below is the structured list of Colab shell commands to check off dependencies, manifests, downloader operations, prototype generation, shape validation, and training runs.

### 1. Install Workspace Requirements
```bash
!pip install torch torchvision pillow sentence-transformers matplotlib
```

### 2. Run Manifest-Driven Image Downloader
Download exactly the whitelisted subset of COCO images to local directories, verifying disjoint splits:
```bash
!python scripts/04_download_images.py --manifest-dir data/processed --image-root data/images
```

### 3. Generate Projected Answer Embedding Prototypes
Generate projected offline class weights using SentenceTransformers (`all-MiniLM-L6-v2`) or our deterministic hash-seeded fallback vector engine:
```bash
!python scripts/build_answer_prototypes.py
```

### 4. Run Shape & Norm Verification Smoke Tests
Programmatically verify batch dimensions, model layers compatibility, prototype unit L2 norms, and type-aware loss loops:
```bash
!python scripts/smoke_test_prototype_model.py
```

### 5. Train Classifier Student Model (`gapcnn_s`, 128x128)
```bash
!python scripts/train_question_vqa.py \
    --head-type classifier \
    --image-encoder gapcnn_s \
    --image-size 128 \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-3 \
    --run-name classifier_student_128 \
    --device cuda \
    --amp
```

### 6. Train Classifier Teacher Model (`mobilenet_v3_large`, 224x224)
```bash
!python scripts/train_question_vqa.py \
    --head-type classifier \
    --image-encoder mobilenet_v3_large \
    --image-size 224 \
    --epochs 20 \
    --batch-size 64 \
    --lr 3e-4 \
    --run-name classifier_teacher_224 \
    --device cuda \
    --amp
```

### 7. Train Prototype Cosine Student Model (`gapcnn_s`, 128x128)
Train student model mapping projected unit prototype cosine alignments scaled by a learnable scale parameter:
```bash
!python scripts/train_question_vqa.py \
    --head-type prototype \
    --image-encoder gapcnn_s \
    --image-size 128 \
    --epochs 20 \
    --batch-size 64 \
    --lr 1e-3 \
    --run-name prototype_student_128 \
    --device cuda \
    --amp \
    --learn-logit-scale
```
