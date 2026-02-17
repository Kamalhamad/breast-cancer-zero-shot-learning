# Breast Cancer Classification using Inductive Zero-Shot Learning

## Overview

This project implements an Inductive Zero-Shot Learning (ZSL) framework for breast cancer histopathology image classification using the BreakHis dataset.

The objective was to evaluate whether semantic embeddings can enable classification of unseen tumor subclasses without labeled training examples.

This project explores how alignment between visual feature representations and semantic embeddings can extend classification capability beyond traditional supervised learning boundaries.

---

## Dataset

- BreakHis dataset  
- 7,900+ histopathological images  
- 8 tumor subclasses  
- Benign and malignant categories  
- Stratified sampling (250 samples per class)  

---

## Methodology

### 1. Feature Extraction

- Pretrained ResNet50 (ImageNet weights)  
- Extracted intermediate and final layer features  
- Combined representations for enhanced abstraction  

### 2. Semantic Embeddings

- Word2Vec embeddings (vector size = 100)  
- Generated embeddings for seen and unseen classes  
- Embedded class labels into a shared semantic space  

### 3. Inductive Zero-Shot Learning Framework

- Logistic Regression classifier trained on seen classes  
- Predicted embedding vectors mapped to closest unseen class  
- Evaluation performed exclusively on unseen classes  

---

## Pipeline Summary

Images  
→ ResNet50 Feature Extraction  
→ Feature Combination  
→ Word2Vec Semantic Embeddings  
→ Logistic Regression Classifier  
→ Embedding Similarity Mapping  
→ Unseen Class Prediction  

---

## Results

- Unseen class classification accuracy (Benign vs Malignant): **60%**  
- Demonstrated generalization capability without exposure to unseen subclasses during training  
- Evaluated using accuracy metrics and confusion matrix analysis  
- Highlighted strengths and limitations of inductive zero-shot transfer in medical imaging  

---

## Tech Stack

- Python  
- TensorFlow / Keras  
- Scikit-learn  
- Gensim (Word2Vec)  
- NumPy  
- Pandas  

---

## Project Structure

```
src/
├── data_processing.py
├── feature_extraction.py
├── embeddings.py
├── train.py
├── zsl_inference.py
└── evaluation.py
```

---

## How to Run

```bash
pip install -r requirements.txt
python src/train.py
python src/zsl_inference.py
```

---

## Key Learnings

- Alignment between visual and semantic embedding spaces  
- Transfer learning for robust feature extraction  
- Modular pipeline design for machine learning workflows  
- Zero-shot inference strategy for unseen class prediction  
- Working under computational constraints  

---

## Future Improvements

- Replace Logistic Regression with deep metric learning  
- Use contextual embeddings (e.g., BERT-based representations)  
- Deploy inference API using FastAPI  
- Perform systematic hyperparameter optimization  
- Explore ensemble methods for improved robustness  
