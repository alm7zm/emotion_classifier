# Facial Expression Recognition (CNN + OpenCV)
## Overview
This project builds a **Convolutional Neural Network (CNN)** to classify human facial expressions and deploys it in a **real-time webcam application**.

The system detects faces using OpenCV and predicts emotions live using a trained deep learning model.
---

## Dataset

This project uses the **FER2013 dataset**, a widely used benchmark for facial expression recognition.

### Dataset Characteristics
- Image size: **48 × 48 pixels**
- Format: **Grayscale images**
- Faces are centered and normalized
- Originally contains **7 emotion classes**

### Original Classes
- Angry  
- Disgust  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

### Preprocessing & Modifications
To improve model performance and handle class imbalance:

- **Removed "Disgust" class** (very few samples)
- Applied **undersampling** to balance all classes
- Converted images to:
  - Shape: `(48, 48, 1)`
  - Normalized pixel values `[0, 1]`

### Final Dataset
- **6 classes used**
- **19,026 training images** (~3,171 per class)
- **4,986 testing images** (~831 per class)

---

## Features

- Real-time facial emotion detection  
- CNN-based classification   
- Preprocessed and balanced dataset   
- Smooth predictions (reduced flickering)  
- Lightweight and efficient model  

---
## Emotion Classes

The model predicts **6 emotions**:

- Angry  
- Fear  
- Happy  
- Sad  
- Surprise  
- Neutral  

---
## Model Details

- Input: **48×48 grayscale images**
- Architecture:
  - 4 Convolutional layers (64 → 128 → 256 → 512)
  - Batch Normalization
  - MaxPooling
  - Dropout
  - Global Average Pooling
  - Dense + Softmax output

---
## Performance

- Training Accuracy: ~95%  
- Validation Accuracy: ~61–62%  
- Test Accuracy: ~61.7%  

### Common Confusions:
- Fear vs Surprise  
- Sad vs Neutral  
- Angry vs Fear  

---
## Real-Time Demo


- Captures webcam input
- Detects face using Haar Cascade
- Preprocesses to 48×48 grayscale
- Predicts emotion using CNN
- Displays emotion label + confidence
- Run Live using: python live_emotion_camera.py

---
## How to Run

### 1. Create Environment
```bash
conda create -n emotion_env python=3.10
conda activate emotion_env
pip install keras opencv-python numpy


