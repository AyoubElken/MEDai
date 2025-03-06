
# Medical Image Classification Projects

This repository contains two distinct projects focused on medical image classification using deep learning techniques. The first project involves classifying kidney CT scan images into four categories (Normal, Cyst, Stone, Tumor), while the second project focuses on detecting brain tumors from MRI images. Both projects leverage state-of-the-art deep learning frameworks (TensorFlow/Keras for the kidney project and PyTorch for the brain tumor project) to achieve high accuracy in medical image classification.

---

## Project 1: Kidney CT Scan Classification (TensorFlow/Keras)

### Overview
This project aims to classify kidney CT scan images into four categories: **Normal**, **Cyst**, **Stone**, and **Tumor**. The model is built using **MobileNetV2** as the base architecture, fine-tuned for the specific task. The dataset consists of 12,446 images, split into training and validation sets.

### Key Features
- **Dataset**: 12,446 kidney CT scan images across 4 classes.
- **Model**: Fine-tuned **MobileNetV2** with additional dense layers for classification.
- **Training**: 15 epochs with Adam optimizer and sparse categorical cross-entropy loss.
- **Performance**: Achieved **100% training accuracy** and **99.12% validation accuracy**.
- **Visualization**: Includes training/validation loss and accuracy plots, as well as a confusion matrix.


### Requirements
- TensorFlow 2.x
- Keras
- NumPy
- Matplotlib
- OpenCV

---

## Project 2: Brain Tumor Detection (PyTorch)

### Overview
This project focuses on detecting brain tumors from MRI images. The dataset consists of 253 images, split into two classes: **Healthy** and **Tumor**. A custom **Convolutional Neural Network (CNN)** is implemented using PyTorch to classify the images.

### Key Features
- **Dataset**: 253 MRI images (155 Tumor, 98 Healthy).
- **Model**: Custom CNN with two convolutional layers, pooling, and fully connected layers.
- **Training**: 400 epochs with Adam optimizer and Binary Cross-Entropy Loss.
- **Performance**: Achieved **100% accuracy** on the validation set.
- **Visualization**: Includes training/validation loss plots and a confusion matrix.


### Requirements
- PyTorch
- NumPy
- Matplotlib
- Scikit-learn
- OpenCV

---

## Documentation References

### Kidney CT Scan Classification
### 1. **TensorFlow and Keras Documentation**
   - **Links**:
     - [TensorFlow Documentation](https://www.tensorflow.org/api_docs)
     - [Keras Documentation](https://keras.io/api/)

---

### 2. **MobileNetV2 Paper**
   - **Title**: "MobileNetV2: Inverted Residuals and Linear Bottlenecks"
   - **Link**: [MobileNetV2 Paper](https://arxiv.org/abs/1801.04381)

---

### 3. **Medical Image Analysis with Deep Learning**
   - This resource provides an overview of deep learning techniques applied to medical imaging, including preprocessing, augmentation, and model evaluation.
   - **Link**: [Medical Image Analysis with Deep Learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6407735/)

---

### 4. **Transfer Learning in Medical Imaging**
   - **Why it's useful**: This resource explains how transfer learning (using pre-trained models like MobileNetV2) can be applied to medical imaging tasks.
   - **Link**: [Transfer Learning for Medical Image Analysis](https://arxiv.org/abs/1902.07208)

---

### 5. **Grad-CAM for Model Interpretability**
   - **Why it's useful**: Grad-CAM (Gradient-weighted Class Activation Mapping) helps visualize which parts of the image the model focuses on, making it useful for medical imaging.
   - **Link**: [Grad-CAM Paper](https://arxiv.org/abs/1610.02391)

---

### 6. **Related Research Papers**
   - **Links**:
     - [Kidney Tumor Classification Using Deep Learning](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7780008/)
     - [Deep Learning for Medical Image Analysis](https://arxiv.org/abs/2004.01608)

---

### 7. **GitHub Repositories for Inspiration**
- **Links**:
     - [Kidney Tumor Segmentation](https://github.com/neheller/kits19)
     - [Medical Image Classification](https://github.com/ozan-oktay/Medical-Image-Classification)
### Brain Tumor Detection
- **PyTorch Documentation**: Official documentation for PyTorch.  
  [Visit PyTorch Docs](https://pytorch.org/docs/stable/index.html)  

- **Scikit-learn Documentation**: Official documentation for Scikit-learn.  
  [Visit Scikit-learn Docs](https://scikit-learn.org/stable/)  


## Contributions
Contributions are welcome! If you have suggestions for improvements or new features, feel free to open an issue or submit a pull request.


## License
Both projects are licensed under the CC0 1.0 Universal. See the [LICENSE](LICENSE) file for details.



