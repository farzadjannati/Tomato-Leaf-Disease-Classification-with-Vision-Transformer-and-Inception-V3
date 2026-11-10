<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=waving&color=0:14532d,50:4ade80,100:ecfccb&height=220&section=header&text=Tomato%20Leaf%20Disease%20Classification&fontSize=38&fontColor=052e16&fontAlignY=50&animation=fadeIn" />
</div>

---

# Tomato Leaf Disease Classification: Vision Transformer versus Inception-V3

> A comparative deep learning study that trains a compact Vision Transformer and an Inception-V3 CNN from scratch to diagnose nine tomato leaf diseases and pests (plus healthy leaves), covering data balancing, model design, evaluation, and patch-embedding interpretability.

<div align="left">

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Torchvision](https://img.shields.io/badge/Torchvision-Transforms_&_Models-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/vision/stable/index.html)
[![TorchMetrics](https://img.shields.io/badge/TorchMetrics-Accuracy-792EE5?style=flat&logo=lightning&logoColor=white)](https://lightning.ai/docs/torchmetrics/stable/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Metrics_&_Splits-F7931E?style=flat&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-Data_Handling-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-11557C?style=flat)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-Confusion_Matrices-4C72B0?style=flat)](https://seaborn.pydata.org/)
[![Inception-V3](https://img.shields.io/badge/CNN-Inception--V3-15803D?style=flat)](https://arxiv.org/abs/1512.00567)
[![ViT](https://img.shields.io/badge/Transformer-Vision_Transformer-0E7490?style=flat)](https://arxiv.org/abs/2010.11929)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Dataset_Hub-F5A623?style=flat&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/ArianFiroozi/NNDL_HW5_S2025)
[![Dataset](https://img.shields.io/badge/Dataset-PlantVillage_Tomato-65A30D?style=flat)](https://www.kaggle.com/datasets/emmarex/plantdisease)
[![Domain](https://img.shields.io/badge/Domain-Plant_Disease_Recognition-059669?style=flat)](#)
[![Colab](https://img.shields.io/badge/Google-Colab-F9AB00?style=flat&logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![License](https://img.shields.io/badge/License-MIT-4B5563?style=flat)](https://opensource.org/licenses/MIT)

</div>

## Abstract

Early and accurate identification of tomato leaf diseases is essential for crop protection, yet manual diagnosis is slow and error-prone. This project compares a convolutional **Inception-V3** network (with an auxiliary classifier) and a compact **Vision Transformer (ViT)** on a PlantVillage-derived dataset of 9,325 images in 10 classes. Both models are trained **from scratch**. Because the dataset is imbalanced (373 to 1,000 images per class), the training split is balanced to 800 images per class using label-safe augmentation, while the validation split stays untouched. Inception-V3 reaches **97.27% validation accuracy** with 26.85M parameters, versus **93.51%** for the ViT with 15.40M parameters.

## Table of Contents

1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Methodology](#methodology)
   - 3.1 [Pipeline](#31-pipeline)
   - 3.2 [Preprocessing and Augmentation](#32-preprocessing-and-augmentation)
   - 3.3 [Inception-V3](#33-inception-v3)
   - 3.4 [Vision Transformer](#34-vision-transformer)
   - 3.5 [Training Setup](#35-training-setup)
4. [Results and Evaluation](#results-and-evaluation)
   - 4.1 [Overall Comparison](#41-overall-comparison)
   - 4.2 [Training Curves](#42-training-curves)
   - 4.3 [Per-Class Performance](#43-per-class-performance)
   - 4.4 [Confusion Matrices](#44-confusion-matrices)
   - 4.5 [ViT Embedding Visualizations](#45-vit-embedding-visualizations)
   - 4.6 [Discussion](#46-discussion)
5. [Tools, Structure, and Installation](#tools-structure-and-installation)
6. [License](#license)
7. [Author](#author)
8. [Support](#support)

---

# Overview

Convolutional networks encode a locality prior through small sliding kernels, while Vision Transformers split an image into patches and learn global relationships through self-attention. This project asks: **on a mid-sized, imbalanced plant disease dataset and without pre-training, which architecture diagnoses tomato leaf diseases better, and at what parameter cost?**

Key features:

* **From-scratch training:** both models are randomly initialized, so the comparison reflects architecture and data efficiency rather than transfer learning.
* **Label-safe balancing:** only horizontal flip, a mild crop (at most 10%), and brightness jitter (at most 10%) are used, preserving the color cues that identify diseases.
* **No leakage:** the stratified split happens before augmentation, and only the training split is augmented.
* **Auxiliary-loss training:** Inception-V3 is optimized with `loss = main + 0.4 x auxiliary`.
* **Custom ViT:** patch embedding, eight residual projection blocks, learned positional embeddings, and a transformer encoder in plain PyTorch.
* **Interpretability:** ViT patch and positional embeddings are rendered as 10x10 maps.

---

# Dataset

The data is a tomato subset of **PlantVillage**, hosted on the Hugging Face Hub as `ArianFiroozi/NNDL_HW5_S2025` (original source: [Kaggle PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease)). It has **9,325 images** in **10 classes**.

| Label | Class | Label | Class |
|:-----:|-------|:-----:|-------|
| 0 | Target Spot | 5 | Leaf Mold |
| 1 | Spider Mites | 6 | Early Blight |
| 2 | Bacterial Spot | 7 | Yellow Leaf Curl Virus |
| 3 | Late Blight | 8 | Healthy |
| 4 | Septoria Leaf Spot | 9 | Mosaic Virus |

<div align="center">
  <img src="assets/01_class_samples.png" alt="One sample leaf image from each class" width="680"/>
  <br>
  <p><em>Figure 1. One sample from each class with its label and name.</em></p>
</div>

Eight classes have 1,000 images, class 5 has 952, and class 9 (Mosaic Virus) has only 373 (about 4%). The data is split 80/20 with stratification (`random_state=42`), giving 7,460 training and 1,865 validation images. Minority classes in the training split are then augmented to 800 images each (8,000 total). Samples are drawn without replacement first, so every unique image is used before any repeat. The validation set stays unbalanced (8 classes x 200, class 5 x 190, class 9 x 75).

<div align="center">
  <img src="assets/02_class_distribution_original.png" alt="Original class distribution" width="400"/>
  <img src="assets/03_class_distribution_balanced.png" alt="Balanced training class distribution" width="400"/>
  <br>
  <p><em>Figure 2. Class distribution before splitting (left) and in the training set after balancing (right).</em></p>
</div>

---

# Methodology

## 3.1 Pipeline

```mermaid
flowchart LR

A["PlantVillage Tomato: 9,325 images"] --> B["Stratified Split 80/20"]
B --> C["Balance Train Split: 800 per class"]
B --> V["Validation Split (untouched)"]
C --> D["Inception Pipeline: 299x299"]
C --> E["ViT Pipeline: 80x80"]
V --> D
V --> E
D --> F["Inception-V3 + Auxiliary Head"]
E --> G["Vision Transformer"]
F --> H["Adam + Cross-Entropy, best checkpoint by val loss"]
G --> H
H --> I["Curves, Confusion Matrix, Per-Class Metrics, Embedding Maps"]
```

## 3.2 Preprocessing and Augmentation

| Stage | Pipeline |
|-------|----------|
| Offline balancing (minority classes only) | `RandomHorizontalFlip(p=0.75)`, `RandomResizedCrop(256, scale=(0.9, 1.0))`, `ColorJitter(brightness=0.1)` |
| Online training | `RandomHorizontalFlip`, `RandomResizedCrop(256, scale=(0.9, 1.0))`, `Resize`, `ToTensor`, `Normalize` |
| Online validation | `Resize`, `ToTensor`, `Normalize` |

Normalization uses ImageNet statistics. Images are resized to **299x299 for Inception-V3** and **80x80 for ViT**, which allows an exact 10x10 grid of 8x8 patches (the paper's 64x64 input cannot produce 100 patches of that size).

## 3.3 Inception-V3

The `torchvision` Inception-V3 is randomly initialized (`pretrained=False`) with `aux_logits=True`, and both heads are replaced for 10 classes.

| Component | Configuration |
|-----------|---------------|
| Main head | `Linear(2048, 1024)`, `ReLU`, `Dropout(0.3)`, `Linear(1024, 10)` |
| Auxiliary head | `Linear(768, 512)`, `ReLU`, `Dropout(0.3)`, `Linear(512, 10)` |
| Loss | `CE(main) + 0.4 x CE(aux)` in training; main output only at inference |
| Parameters | **26.85M** |

## 3.4 Vision Transformer

```mermaid
flowchart TD

A["Input 80x80x3"] --> B["Patchify: 100 patches of 8x8x3"]
B --> C["Flatten 100 x 192, Linear 192 to 64"]
C --> D["8 Projection Blocks: 64-128-64, Residual, LayerNorm"]
D --> E["Add Learned Positional Embedding"]
E --> F["Transformer Encoder: 2 heads, FFN 64-128-64"]
F --> G["Flatten 6400, MLP 2048-1024-10"]
G --> H["Class Logits"]
```

| Component | Configuration |
|-----------|---------------|
| Patch embedding | `Linear(192, 64)` on 100 patches |
| Projection blocks | 8 blocks, `64 -> 128 -> 64`, residual, `LayerNorm`, dropout 0.1 |
| Positional embedding | `nn.Embedding(100, 64)`, `trunc_normal_` init (`std=0.02`) |
| Transformer encoder | 1 block, 2 attention heads |
| Classifier | `6400 -> 2048 -> 1024 -> 10`, ReLU, dropout 0.3 |
| Parameters | **15.40M** |

## 3.5 Training Setup

| Setting | Inception-V3 | Vision Transformer |
|---------|--------------|--------------------|
| Input / batch size | 299x299 / 32 | 80x80 / 256 |
| Optimizer | Adam, `lr=1e-4`, `eps=1e-7`, no weight decay | Same |
| Epochs | 30 | 30 |
| Loss | CE + 0.4 x auxiliary CE | CE |
| Time per epoch | ~47 s | ~4-7 s |
| Model selection | Lowest validation loss | Lowest validation loss |

The Inception-V3 learning rate was chosen with a 4-epoch sweep over `{0.1, 0.01, 0.001, 0.0001}`, which gave 9.7%, 30.3%, 84.4%, and 86.5% training accuracy respectively.

---

# Results and Evaluation

## 4.1 Overall Comparison

| Model | Parameters | Input | Validation Accuracy | Macro F1 |
|-------|-----------:|-------|--------------------:|---------:|
| **Inception-V3** | 26.85M | 299x299 | **97.27%** | **0.973** |
| Vision Transformer | 15.40M | 80x80 | 93.51% | 0.936 |

## 4.2 Training Curves

<div align="center">
  <img src="assets/04_inception_training_curves.png" alt="Inception-V3 curves" width="440"/>
  <img src="assets/06_vit_training_curves.png" alt="ViT curves" width="440"/>
  <br>
  <p><em>Figure 3. Loss and accuracy over 30 epochs for Inception-V3 (left) and the ViT (right).</em></p>
</div>

At epoch 30, Inception-V3 reaches about 97.5% training and 97.3% validation accuracy. The ViT reaches about 98.3% training but 93.5% validation accuracy, a clearly larger generalization gap.

## 4.3 Per-Class Performance

| Class | Name | Support | Inception F1 | ViT F1 | Inception Recall | ViT Recall |
|:-----:|------|--------:|-------------:|-------:|-----------------:|-----------:|
| 0 | Target Spot | 200 | 0.9645 | 0.9044 | 0.9500 | 0.8750 |
| 1 | Spider Mites | 200 | 0.9681 | 0.9291 | 0.9850 | 0.9500 |
| 2 | Bacterial Spot | 200 | 0.9824 | 0.9363 | 0.9750 | 0.9550 |
| 3 | Late Blight | 200 | 0.9698 | 0.9114 | 0.9650 | 0.9000 |
| 4 | Septoria Leaf Spot | 200 | 0.9657 | 0.9435 | 0.9850 | 0.9600 |
| 5 | Leaf Mold | 190 | 0.9869 | 0.9468 | 0.9895 | 0.9368 |
| 6 | Early Blight | 200 | 0.9482 | 0.8901 | 0.9150 | 0.8500 |
| 7 | Yellow Leaf Curl Virus | 200 | 0.9728 | 0.9563 | 0.9850 | 0.9850 |
| 8 | Healthy | 200 | 0.9925 | 0.9875 | 0.9950 | 0.9850 |
| 9 | Mosaic Virus | 75 | 0.9804 | 0.9548 | 1.0000 | 0.9867 |

## 4.4 Confusion Matrices

<div align="center">
  <img src="assets/05_inception_confusion_matrix.png" alt="Inception-V3 confusion matrix" width="400"/>
  <img src="assets/07_vit_confusion_matrix.png" alt="ViT confusion matrix" width="400"/>
  <br>
  <p><em>Figure 4. Validation confusion matrices for Inception-V3 (left) and the ViT (right).</em></p>
</div>

## 4.5 ViT Embedding Visualizations

<div align="center">
  <img src="assets/08_vit_patch_embedding_class0.png" alt="Patch embedding map, class 0" width="520"/>
  <img src="assets/09_positional_encoding.png" alt="Learned positional embedding" width="300"/>
  <br>
  <p><em>Figure 5. Left: image and patch-embedding dimension 0 on the 10x10 grid (class 0; all ten classes are in <code>assets/</code>). Right: learned positional embedding (mean over 64 dimensions) upsampled to 80x80.</em></p>
</div>

## 4.6 Discussion

* **Accuracy:** Inception-V3 is about 3.8 points more accurate and leads on every class. Both models find Healthy easiest and Early Blight hardest.
* **Efficiency:** the ViT has about 43% fewer parameters and trains roughly 10x faster per epoch, which suits resource-limited settings if a small accuracy loss is acceptable.
* **Data efficiency:** the convolutional locality prior helps when training from scratch, whereas the ViT must learn spatial structure itself and overfits more.
* **Caveats:** the models use different input resolutions (299 vs. 80), which may account for part of the gap. The validation set is also used for checkpoint selection and there is no separate test set, so the figures are slightly optimistic.

---

# Tools, Structure, and Installation

**Stack:** Python 3.10+, PyTorch, Torchvision, TorchMetrics, scikit-learn, Pandas, PyArrow, NumPy, Pillow, Matplotlib, Seaborn, Hugging Face Hub, Git LFS, Google Colab.

```text
Tomato-Leaf-Disease-Classification-with-Vision-Transformer-and-Inception-V3
│
├── tomato_leaf_disease_vit_vs_inception.ipynb   # Main implementation notebook
├── assets/                                      # Figures exported from notebook outputs
├── requirements.txt                             # Dependencies
├── LICENSE
└── README.md
```

```bash
# 1. Clone the repository
git clone https://github.com/farzadjannati/Tomato-Leaf-Disease-Classification-with-Vision-Transformer-and-Inception-V3.git
cd Tomato-Leaf-Disease-Classification-with-Vision-Transformer-and-Inception-V3

# 2. Create and activate a virtual environment
conda create -n tomato-vit python=3.10
conda activate tomato-vit

# 3. Install required packages
pip install -r requirements.txt
# Core deps: torch, torchvision, torchmetrics, scikit-learn, pandas, pyarrow, seaborn, matplotlib, tqdm, pillow

# 4. Download the dataset (requires Git LFS)
git lfs install
git clone https://huggingface.co/datasets/ArianFiroozi/NNDL_HW5_S2025
```

**Reproducibility:** run the notebook cells sequentially. A CUDA GPU is strongly recommended (for example a Colab GPU runtime). The split and balancing use seed `42`; weights are randomly initialized, so accuracy can vary slightly between runs.

---
# License

This project is licensed under the MIT License.

---

## Author

**Farzad Jannati**
M.Sc. Student, University of Tehran
Research Assistant @ Social Networks Lab

**Research Interests:** NLP, Large Language Models (LLMs), Vision-Language Models, Adversarial Robustness, Parameter-Efficient Fine-Tuning, Agentic AI, Retrieval-Augmented Generation (RAG)

📧 [farzadjannati@ut.ac.ir](mailto:farzadjannati@ut.ac.ir) | 💻 [github.com/farzadjannati](https://github.com/farzadjannati) | 💼 [linkedin.com/in/farzadjannati](https://www.linkedin.com/in/farzadjannati)

---

# Support

If you find this project useful, consider giving it a star ⭐

---

<p align="center">
  Built with ❤️ using PyTorch, Torchvision, TorchMetrics, and Scikit-learn
</p>
