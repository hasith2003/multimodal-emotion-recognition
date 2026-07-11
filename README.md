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

The project has been refactored from Jupyter Notebooks into a modular Python architecture under the `src/` directory:

* **[src/config.py](file:///d:/Emotion/src/config.py)**: Configuration variables including dataset locations, output directories, label mappings, and hyperparameters.
* **[src/extract_features.py](file:///d:/Emotion/src/extract_features.py)**: Module for processing and pre-extracting audio (Wav2Vec2) and video (MTCNN + FAb-Net) features.
* **[src/dataset.py](file:///d:/Emotion/src/dataset.py)**: Custom PyTorch dataset reader (`MeldDataset`) and collate/padding logic.
* **[src/models.py](file:///d:/Emotion/src/models.py)**: Network components (`IMABlock` cross-attention and `Trimodal_SSE_FT` fusion network).
* **[src/train.py](file:///d:/Emotion/src/train.py)**: Main CLI script to execute training, validation, and evaluation loops.
* **[src/check_pipeline.py](file:///d:/Emotion/src/check_pipeline.py)**: Integration dry-run verification script.

---

## Setup & Running the Pipeline

Ensure all dependencies are installed (e.g. PyTorch, OpenCV, Transformers, facenet-pytorch). 

### 1. Extract Embeddings (Pre-extraction)
Run the feature extraction script to compute facial and audio embeddings for the raw video datasets:
```bash
# Extract both audio and video embeddings
python src/extract_features.py --modality all

# Extract audio only
python src/extract_features.py --modality audio

# Extract video only
python src/extract_features.py --modality video
```

### 2. Verify Pipeline Integration (Dry-run)
Run the check script to run a quick test iteration over model and dataset loaders:
```bash
python src/check_pipeline.py
```

### 3. Training & Evaluation
To train a model from scratch:
```bash
python src/train.py --epochs 5 --batch_size 8 --lr 2e-5
```

To run validation and test evaluation only using a saved checkpoint path:
```bash
python src/train.py --eval_only --model_path trimodal_emotion_model.pth
```

---

## Evaluation Results

The trimodal emotion model trained with standard CrossEntropy Loss (`trimodal_emotion_model_epoch10.pth`) achieved the following performance metrics:

* **Validation (DEV) Set Accuracy**: **51.71%**
* **Test Set Accuracy**: **56.86%**

### Detailed Class-wise Test Performance

| Emotion | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **neutral** | 0.59 | 0.95 | 0.72 | 1256 |
| **joy** | 0.55 | 0.28 | 0.37 | 402 |
| **sadness** | 0.34 | 0.09 | 0.14 | 208 |
| **anger** | 0.45 | 0.36 | 0.40 | 345 |
| **surprise** | 0.85 | 0.15 | 0.25 | 281 |
| **fear** | 0.00 | 0.00 | 0.00 | 50 |
| **disgust** | 1.00 | 0.01 | 0.03 | 68 |
| **Accuracy** | | | **0.57** | **2610** |

### Detailed Class-wise Validation (DEV) Performance

| Emotion | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **neutral** | 0.54 | 0.91 | 0.68 | 469 |
| **joy** | 0.49 | 0.29 | 0.37 | 163 |
| **sadness** | 0.38 | 0.19 | 0.25 | 111 |
| **anger** | 0.43 | 0.35 | 0.38 | 153 |
| **surprise** | 0.74 | 0.15 | 0.25 | 150 |
| **fear** | 0.00 | 0.00 | 0.00 | 40 |
| **disgust** | 0.00 | 0.00 | 0.00 | 22 |
| **Accuracy** | | | **0.52** | **1108** |

---

## License

This project is licensed under the MIT License - see the [LICENSE](file:///d:/Emotion/LICENSE) file for details.

