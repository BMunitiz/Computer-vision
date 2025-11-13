# Garbage Classification Project

## Overview
This Jupyter Notebook implements a garbage classification system using deep learning with TensorFlow/Keras. The model is designed to classify images of waste items into 30 different categories for recycling and waste management purposes.

## Project Structure
The notebook contains the following main components:

### 1. Environment Setup
- Installation of required dependencies:
  - `pydot` for model visualization
  - `tensorflow-macos` for TensorFlow on macOS
  - `tensorflow-metal` for GPU acceleration on Apple Silicon
- GPU availability check using TensorFlow

### 2. Data Loading and Preprocessing
- Uses Keras `image_dataset_from_directory` to load images
- Dataset split: 70% training (10,500 images) and 30% validation (4,500 images)
- Image size: 180x180 pixels
- Batch size: 128
- 30 different garbage categories

### 3. Data Exploration
- Visualization of the first 9 images from the dataset
- Display of class names including:
  - Various plastic items (bottles, bags, containers)
  - Glass containers
  - Paper products
  - Metal cans
  - Organic waste
  - Textiles and other materials

## Key Features
- **Multi-class Classification**: 30 different garbage categories
- **Data Augmentation**: Built-in image preprocessing
- **GPU Acceleration**: Optimized for macOS with Metal support
- **Transfer Learning Ready**: Structure compatible with pre-trained models

## Dataset
The dataset contains 15,000 images organized into 30 garbage categories, making it suitable for training robust waste classification models.

## Requirements
- Python 3.11
- TensorFlow 2.16.2
- Keras 3.6.0
- Additional dependencies listed in the installation cells

## Usage
1. Ensure all dependencies are installed
2. Verify GPU availability (if using macOS with M1 chip)
3. Run the cells sequentially to:
   - Load and preprocess the data
   - Explore the dataset
   - Train the classification model
   - Evaluate model performance

## Note
The notebook sets up the foundation for building and training a convolutional neural network for waste categorization, which is crucial for recycling automation and waste management systems.
