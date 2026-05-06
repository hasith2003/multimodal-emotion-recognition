# Multimodal Emotion Recognition
## Transformer-Based Self-Supervised Feature Fusion

This repository provides an implementation of multimodal emotion recognition using the MELD dataset. The approach leverages Transformer architectures to perform self-supervised feature fusion between audio and video modalities.

## Training Pipeline

The project is implemented in two main stages:

### 1. Feature Extraction (Embeddings)
To make training efficient, we first extract high-dimensional embeddings:
* **Video:** Facial features are extracted and embedded using self-supervised models.
* **Audio:** Audio signals are processed into rich feature vectors.
These embeddings are stored to avoid redundant computation during the model training phase.

### 2. Trimodal Model Training

The second stage involves feeding the pre-extracted audio and video embeddings, along with the corresponding textual transcripts which are converted to embeddings by RoBERTa, into the Transformer-based fusion model. The model utilizes inter modality attention mechanisms to learn and weight the most relevant features across all three modalities simultaneously, enabling more robust emotion prediction by capturing the nuances of each data stream.

## Custom Focal Loss for MELD
The MELD dataset contains 7 classes with a significant **class imbalance** (e.g., a high number of 'Neutral' samples compared to 'Disgust' or 'Fear').

While standard **CrossEntropy Loss** often leads the model to over-predict majority classes, I introduced a **Custom Focal Loss** implementation. 

### Why Focal Loss?
Focal Loss down-weights the loss contributed by easy-to-classify examples and focuses the training on hard, misclassified samples. This ensures the model achieves higher accuracy across **all 7 classes**, specifically improving the performance on minority categories.

## Project Structure
- `Emotion.ipynb`: The primary notebook for model training and evaluation.
- `.gitignore`: Configured to exclude raw datasets and large `.pth` model weights.

## Setup
1. Extract your embeddings using the provided Python scripts.
2. Training the fusion model via the `Emotion.ipynb` notebook.
