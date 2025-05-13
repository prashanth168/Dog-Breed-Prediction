
# 🐕 Dog Breed Classification using InceptionV3

This project involves using deep learning techniques to classify dog breeds from images. It leverages the power of the InceptionV3 model for feature extraction and custom layers for classification. The model has been trained on a dataset of 70 dog breeds with performance evaluated using validation accuracy.

## 📊 Model Performance
- **Training Accuracy:** 96.50%
- **Validation Accuracy:** 91.67%
- **Validation Loss:** 1.9452
- **Epochs:** 80 (Early stopping applied)

## 🧠 Model Architecture
- **Base Model:** InceptionV3 (pretrained on ImageNet)
- **Custom Layers:**
  - `GlobalAveragePooling2D`
  - `Dense(256 units, ReLU)`
  - `Dropout(0.5)`
  - `Dense(70 units, Softmax)` (for 70 dog breeds)

## 🗂 Dataset Details
- **Classes:** 70 different dog breeds
- **Image Input Shape:** Resized to 299x299 pixels (required by InceptionV3)
- **Preprocessing Steps:**
  - Rescaling
  - Optional data augmentation
  - Label encoding for breed labels

## 🧪 Training Details
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Metrics:** Accuracy
- **Early Stopping:** Monitored `val_loss` with a specified patience
- **Batch Size:** *(Optional - add if known, e.g., 32)*

## 🛠️ Requirements
```bash
tensorflow>=2.x
numpy
matplotlib
scikit-learn
pandas
