# Brain Tumor Classification from MRI Images using Deep Learning

This project presents a deep learning-based system for the automatic classification of brain tumors from MRI scans. It categorizes medical images into four classes:
- Glioma
- Meningioma
- Pituitary Tumor
- No Tumor

Two modeling approaches were implemented:
1. A custom Convolutional Neural Network (CNN) trained from scratch
2. A fine-tuned EfficientNetB0 model using transfer learning

---

## Dataset

The dataset consists of 7,023 brain MRI images, categorized into four classes. The data was split into:
- Training set: 5,712 images
- Test set: 1,311 images (further divided into validation and test subsets)

Images were loaded using `ImageDataGenerator` with appropriate preprocessing for each model.

---

## Tools and Libraries

- Python, TensorFlow, Keras, Pandas, NumPy
- EfficientNetB0 (pre-trained on ImageNet)
- Matplotlib, Seaborn for visualizations
- Google Colab for training and evaluation

---

## Model Architectures

### 1. CNN from Scratch
- Two convolutional layers with max pooling
- Fully connected layers with dropout
- Final softmax layer for classification
- Test Accuracy: ~95.4%

### 2. EfficientNetB0 with Transfer Learning
- Pre-trained base with fine-tuning of the last 40 layers
- Added custom dense layers for classification
- Test Accuracy: ~98.3%

---

## Evaluation Results

### CNN
- Training Accuracy: 99.8%
- Validation Accuracy: 96.6%
- Test Accuracy: 95.4%
- F1-score (macro): ~0.95

### EfficientNetB0
- Training Accuracy: 99.8%
- Validation Accuracy: 98.0%
- Test Accuracy: 98.3%
- F1-score (macro): ~0.98

Evaluation metrics include confusion matrices, classification reports, and accuracy curves over epochs.
