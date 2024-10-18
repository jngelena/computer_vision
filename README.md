# Emotion Detection using Computer Vision

This project focuses on distinguishing between human and pet emotions using computer vision techniques. The aim is to build a system that can classify emotions for both humans and animals using Convolutional Neural Networks (CNNs), specifically based on the EfficientNetB5 architecture.

## Project Inspiration
This project draws inspiration from the paper "Pre-Trained Multi-Modal Transformer for Pet Emotion Detection," which highlights the gap in research on animal emotion recognition. The primary goal is to extend emotion detection methods typically used for humans to also apply to pets, such as cats and dogs.

## Dataset
- **Human Dataset**: We used the FER-2013 dataset, containing 28,709 grayscale images (48x48 pixels) divided into 7 classes: angry, disgust, fear, happy, neutral, sad, and surprise.
- **Pet Dataset**: Contains 1,000 images divided into 4 classes: angry, happy, sad, and other.
- **Combined Dataset**: More than 2,000 RGB images with dimensions around 1000x1000 pixels, split as follows:
  - Training dataset: 1,421 images
  - Validation dataset: 365 images
  - Testing dataset: 244 images

## Model Architecture
The project utilizes a CNN model based on the EfficientNetB5 architecture for image classification. Key elements include:
- Sequential model with batch normalization
- Two dense layers with ReLU activations
- Dropout layer for regularization
- Final dense layer with softmax activation for multi-class classification

## Training
- **Human vs. Pet Classification**:
  - Batch size: 16
  - Epochs: 10
  - Training Accuracy: 100%
  - Test Accuracy: 100%
- **Pet Emotion Detection**:
  - Batch size: 16
  - Epochs: 100
  - Training Accuracy: 96%
  - Test Accuracy: 88%
- **Human Emotion Detection**:
  - Epochs: 30
  - Training Accuracy: 88%
  - Test Accuracy: 63%

## Results
### Human vs. Pet Classification
- **Precision, Recall, F1-score**: Perfect scores (1.00) for both classes.

### Pet Emotion Detection
- Achieved high accuracy in classifying emotions, with an average F1-score across different classes:
  - Angry: 0.93
  - Happy: 0.95
  - Sad: 0.95
  - Other: 0.90

### Human Emotion Detection
Achieved moderate accuracy with room for improvement in recognizing specific emotions like fear and sadness.

## Tools and Libraries
- **Programming Language**: Python
- **Frameworks**: TensorFlow, Keras
- **Data Augmentation**: Keras ImageDataGenerator
- **Visualization**: Training history plotted for better understanding of model performance

## Demo
The project includes a demo built with Streamlit through app.py, showcasing examples of the emotion detection results.
