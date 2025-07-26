# 🍱 Snackly – Malaysian Food Recognition & Nutrition Estimation App

### 📊 Dataset
[📂 Malaysia Food-11 Dataset on Kaggle](https://www.kaggle.com/datasets/karkengchan/malaysia-food-11?resource=download)  
### 🔗 Baseline Reference Model
[🔗 EHFRNet GitHub Repository (Backbone Inspiration)](https://github.com/LduIIPLab/CVnets)

---

## 📌 Overview

**Snackly** is a lightweight AI-powered mobile app that recognizes Malaysian food items from images and estimates their nutritional values. It uses a hybrid deep learning model called **HECTNet**, inspired by EHFRNet, and combines both semantic and handcrafted features for robust classification in real-world conditions.

This project is designed to support health-conscious users in their dietary tracking journey and is especially useful in local contexts with food like **Nasi Lemak**, **Roti Canai**, **Satay**, etc.

---

## 🧠 HECTNet – Hybrid Efficient CNN-Transformer Network

HECTNet is a custom-built model tailored for mobile-friendly food recognition. It consists of:

- **Main Branch**:
  - **Backbone**: MobileNetV2 + LP-ViT (Location-Preserving ViT)
  - Extracts **semantic** global context from food images
- **Auxiliary Branch**:
  - Gabor-based handcrafted texture features across multiple scales
  - Captures **fine-grained** color and texture variations (crispy, saucy, etc.)
- **Bidirectional Cross-Attention Fusion** ("Aha! Moment"):
  - Aligns and fuses the two distinct feature types
  - Allows dynamic, context-aware feature prioritization
- **Classifier**:
  - Final fused vector (160D) passed into a fully connected layer → softmax prediction

> 💡 **“Aha! Moment”** is the coined term describing the model’s critical moment of insight during fusion—where it contextually understands which food-identifying cues matter most.

---

## 🍽️ Target Food Categories

| Class         | Description                       |
|---------------|-----------------------------------|
| Nasi Lemak    | Rice dish with sambal and anchovies |
| Roti Canai    | Flatbread served with dhal        |
| Satay         | Skewered grilled meat             |
| Kaya Toast    | Toasted bread with kaya spread    |
| Fried Rice    | Classic Malaysian-style fried rice |

---

## 📱 Mobile App (Frontend – Flutter)

- **Framework**: Flutter (cross-platform)
- **Features**:
  - Camera input or gallery upload
  - Displays top-1 food label and calorie count
  - Meal logging system (stored via Supabase)
  - Clean UI tailored for Malaysian user base

---

## ⚙️ Backend (FastAPI Inference Server)

- **Framework**: FastAPI + Uvicorn
- **Purpose**:
  - Serve the HECTNet model
  - Accept image input via POST request
  - Return predicted food label and nutritional info
- **Integrated With**:
  - Supabase for authentication and food logging
  - Optional image embedding storage for retrieval

---

## 🗃️ Dataset

- 📦 **Malaysia Food-11** (Kaggle):  
  https://www.kaggle.com/datasets/karkengchan/malaysia-food-11?resource=download
- Images resized to **256x256**
- Data augmentation applied:
  - Random rotation, flips, brightness/contrast shifts
- Train/Validation Split: **80% / 20%**

---

## 🧪 Training Pipeline

- Framework: **PyTorch**
- Loss Function: **CrossEntropyLoss**
- Optimizer: **Adam**
- Evaluation Metrics:
  - Accuracy
  - Precision, Recall, F1-score
  - AUC-ROC
- Epochs: 100 with early stopping
- Embedding Dim: 160D fused features

---

## 📌 Key Technical Innovations

### ✅ Bidirectional Cross-Attention Fusion
- Mutual interaction between main (CNN-ViT) and auxiliary (handcrafted) embeddings
- Achieves better feature complementarity

### ✅ High Performance on Challenging Datasets
- Robust to:
  - High intra-class variation (e.g., nasi lemak with/without egg)
  - Low inter-class distinctiveness (e.g., fried rice vs. nasi lemak)

### ✅ Dual Embedding Use
- Final fused 160D vector is also suitable for:
  - Visual search (content-based image retrieval)
  - Similar food recommendation

---

## 🚀 How to Run Locally

Make sure you have:

- Python 3.8+
- Flutter installed (`flutter doctor`)
- A working Chrome browser for web preview

### Step 1: Clone the Repository

```bash
git clone https://github.com/ihaterynn/HECT-Net.git
```
### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```
### Step 3: HECT-Net Backend Setup

```bash
cd backend
python hectnet_server.py
```

### Step 4: Start the Flutter App

```bash
cd frontend
flutter run -d chrome
```
---

## 📝 License

This project is licensed under the MIT License.

MIT License

Copyright (c) 2025 Ryan Chan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

### 📎 Acknowledge Upstream:

```md
> ⚠️ This project builds upon and significantly extends the [EHFRNet architecture](https://github.com/LduIIPLab/CVnets), originally developed by [Guorui Sheng](https://github.com/GuoruiSheng).  
> Licensed under the MIT License.
