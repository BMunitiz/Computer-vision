# Garbage Classification with Deep Learning

This project implements a multi-class image classification system to identify 30 different types of garbage items using deep learning. The goal is to automate waste sorting for recycling applications.

## 📌 Overview

The system uses three different neural network architectures:

1. **Custom CNN** – Built from scratch with residual connections and separable convolutions.
2. **ResNet50** – Transfer learning using a pre-trained ResNet50 model (frozen, only new classifier trained).
3. **ResNet152** – Transfer learning with the deeper ResNet152 model for improved feature extraction.

The models are trained on the **Garbage Classification Dataset** containing 30 categories of waste items (plastic, paper, glass, metal, organic, etc.).

## 🗂️ Dataset

- **Source**: [Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- **Classes**: 30 (e.g., plastic bottles, cardboard boxes, glass jars, food waste, tea bags, etc.)
- **Total images**: 15,000
- **Split**: 70% training (10,500 images), 30% validation (4,500 images)
- **Image sizes**: 
  - Custom CNN & ResNet50: 180×180
  - ResNet152: 224×224

## 🧠 Models & Approach

### 1. Custom CNN
- Residual blocks with separable convolutions
- Progressive filter increase: 128 → 256 → 512 → 728 → 1024
- Batch normalization, ReLU activations, dropout (0.5)
- Output: 30-class logits (softmax applied at inference)
- **Trainable parameters**: 2.75M

### 2. ResNet50 (Transfer Learning)
- Pre-trained on ImageNet (frozen)
- Global average pooling + Dense(30, softmax)
- **Trainable parameters**: 61,470

### 3. ResNet152 (Transfer Learning)
- Deeper pre-trained ResNet152 (frozen)
- Global average pooling + Dense(30, softmax)
- **Trainable parameters**: 61,470

### Training Configuration
- **Optimizer**: Adam
- **Initial learning rate**: 0.001
- **Learning rate scheduling**: ReduceLROnPlateau (factor 0.2, patience 3)
- **Early stopping**: patience 5
- **Loss**: Categorical cross-entropy (from_logits for custom model)
- **Data augmentation**: random flips, brightness, contrast

## 📊 Results

| Model | Validation Accuracy | Trainable Params | Training Time/epoch |
|-------|---------------------|------------------|---------------------|
| Custom CNN | 76.8% | 2,752,502 | ~530 sec |
| ResNet50 | **82.2%** | 61,470 | ~250 sec |
| ResNet152 | **84.4%** | 61,470 | ~355 sec |

**Key takeaways**:
- Transfer learning dramatically improves performance with far fewer trainable parameters.
- ResNet152 slightly outperforms ResNet50, showing deeper features help distinguish visually similar classes.
- The remaining errors are mainly between similar materials (e.g., cardboard vs paper, metal cans).

## ⚙️ Requirements

Install the dependencies with:

```bash
pip install tensorflow keras numpy matplotlib scikit-learn seaborn
```

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/garbage-classification.git
   cd garbage-classification
   ```

2. Download the dataset and place it in `archive/images/` (or adjust paths in the notebook).

3. Launch Jupyter Notebook:
   ```bash
   jupyter notebook Garbage_classifier_enhanced.ipynb
   ```

4. Run the cells sequentially to:
   - Load and augment data
   - Build and train the custom CNN
   - Load and train ResNet50 and ResNet152
   - Evaluate models with classification reports and confusion matrices
   - Test inference on a sample image

## 📈 Evaluation

- **Classification Reports**: Precision, recall, F1-score per class.
- **Confusion Matrices**: Visualize misclassifications.
- **Training Curves**: Accuracy and loss over epochs.



## 🙏 Acknowledgements

- Dataset: [Garbage Classification Dataset](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification)
- Keras Applications for pre-trained ResNet models.

---

**Note**: The notebook (`Garbage_classifier_enhanced.ipynb`) contains the full implementation, including data loading, augmentation, model definition, training, evaluation, and inference.
