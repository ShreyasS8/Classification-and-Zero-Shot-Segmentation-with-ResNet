# ResNet for Butterfly Classification and Zero-Shot Segmentation

This repository contains the PyTorch implementation of a custom Residual Network (ResNet) for classifying **100 species** of butterflies and moths, plus a **training-free zero-shot segmentation** pipeline that leverages the classifier's gradients to produce masks.

---

## Key Achievements
- **Classification Accuracy:** 96.2% on a 100-class dataset.  
- **Segmentation (zero-shot):** mIoU = 0.796 using gradient-based saliency + GrabCut.

---

## Contents / Highlights
- Custom ResNet-14 (6n+2, n=2) implemented in PyTorch.
- Optimized training with label smoothing, cosine-annealing LR, and mixed-precision.
- Extensive data augmentation (flips, rotations, color-jitter).
- Zero-shot segmentation: compute gradient of predicted class logit w.r.t. input → saliency → automated seeds → GrabCut → final mask.
- Reproducible (seeded experiments).

---

## Architecture (short)
- Input: **224×224×3** images.
- Layer 1: 3×3 conv.
- Stage 1: 2 residual blocks, 32 filters (output 224×224).  
- Stage 2: 2 residual blocks, 64 filters (first block stride=2, output 112×112).  
- Stage 3: 2 residual blocks, 128 filters (first block stride=2, output 56×56).  
- Global Average Pooling → Fully Connected (100 units).

---

## Zero-Shot Segmentation Pipeline (summary)
1. Run trained ResNet-14 on image, get predicted class logit.  
2. Compute gradient of that logit w.r.t. the input image pixels.  
3. Produce a saliency map from gradient magnitude.  
4. Use quantile thresholding to pick **definite foreground** and **definite background** seeds.  
5. Run GrabCut with those seeds.  
6. Output final segmentation mask.

---

## Dataset
- **Dataset:** Butterfly & Moths Image Classification (100 classes).  
- **Train images:** ~12,000 | **Val:** 500 | **Test:** 100.  
- Ground-truth segmentation masks were available for validation/testing (used to compute mIoU).

---

## Results (example)
- Classification accuracy: **96.2%**  
- Segmentation mIoU: **0.796**

---

## Example images
Below are the images included with this README (placed in `images/`):

**Brookes Birdwing (original)**  
![Brookes Birdwing](images/brookes_birdwing.jpg)

**Brookes Birdwing (RGB / processed)**  
![Brookes Birdwing RGB](images/brookes_birdwing_rgb.jpg)

**African Giant Swallowtail (original)**  
![African Giant Swallowtail](images/african_giant_swallowtail.jpg)

**African Giant Swallowtail (RGB / processed)**  
![African Giant Swallowtail RGB](images/african_giant_swallowtail_rgb.jpg)

**Atala (original)**  
![Atala](images/atala.jpg)

**Atala (RGB / processed)**  
![Atala RGB](images/atala_rgb.jpg)

> Note: When viewing this README on GitHub, ensure the `images/` folder is included in the repo at the same level as `README.md` so the images render correctly.

---

## Setup & Installation (quick)
```bash
git clone https://github.com/your-username/resnet-zeroshot-segmentation.git
cd resnet-zeroshot-segmentation
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` should include at least:
```
torch
torchvision
numpy
opencv-python
Pillow
```

---

## Usage

### Train
```bash
python train.py <path_to_train_data_dir> <path_to_model_output_dir>
# Example:
python train.py ./dataset/train ./checkpoints
```

### Evaluate & Generate Segmentation Masks
```bash
python evaluate.py <path_to_model_ckpt> <path_to_test_imgs_dir>
# Example:
python evaluate.py ./checkpoints/resnet_model.pth ./dataset/test
```

Outputs produced by `evaluate.py`:
- `submission.csv` — predicted class for each test image.  
- `seg_maps/` — generated segmentation masks (one per test image).


## Reproducibility
- All experiments in the original repository were seeded. For exact replication, set seeds for `random`, `numpy`, and `torch` and use deterministic flags as needed.

---

## Attribution
Images in `images/` were supplied by the project owner and embedded here for demonstration purposes.

---

If you'd like, I can also:
- produce a `README.zip` containing the `README.md` and the `images/` folder, or
- create a full repository structure with placeholders and push a ZIP for download.

Let me know which option you prefer.
