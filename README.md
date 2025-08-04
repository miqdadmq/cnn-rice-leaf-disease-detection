# 🌾 Rice Leaf Disease Detection using CNN

This project detects diseases in rice leaves using a Convolutional Neural Network (CNN). It helps farmers identify plant diseases early using image classification.

## 📂 Dataset
- 3 classes:
  - Healthy
  - Bacterial leaf blight
  - Brown spot
- Image size: 224x224 (resized)

## 🧠 Model
- CNN with 3 convolutional layers
- Optimizer: Adam
- Loss: Categorical Crossentropy
- Accuracy: ~92% on test set

## 🚀 How to Run
```bash
git clone https://github.com/namakamu/rice-leaf-detection-cnn.git
cd rice-leaf-detection-cnn
pip install -r requirements.txt
python train.py
