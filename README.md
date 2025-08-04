# 🌾 Rice Leaf Disease Detection using CNN

This project detects diseases in rice leaves using a Convolutional Neural Network (CNN). It helps farmers identify plant diseases early using image classification.

## 📂 Dataset
- 5 classes:
  - bacterial_blight
  - blast
  - brownspot
  - healthy
  - tungro
- Image size: 180x180 (resized)

## 🧠 Model
- CNN with 3 convolutional layers
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Accuracy: ~98% on test set

## 🚀 How to Run
```bash
git clone https://github.com/miqdadmq/cnn-rice-leaf-disease-detection.git
cd cnn-rice-leaf-disease-detection
pip install -r requirements.txt
python train.py
