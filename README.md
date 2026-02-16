Breast Cancer Classification using Inductive Zero-Shot Learning
Overview

This project implements an Inductive Zero-Shot Learning (ZSL) framework for breast cancer histopathology image classification using the BreakHis dataset.

The objective was to evaluate whether semantic embeddings can enable classification of unseen tumor subclasses without labeled training examples.

Dataset

BreakHis dataset

7,900+ histopathological images

8 tumor subclasses

Benign and malignant categories

Stratified sampling (250 samples per class)

Methodology
1. Feature Extraction

Pretrained ResNet50 (ImageNet weights)

Extracted both intermediate and final layer features

Combined representations for enhanced abstraction

2. Semantic Embeddings

Word2Vec embeddings (vector size = 100)

Generated embeddings for both seen and unseen classes

Embedded class labels into shared semantic space

3. Inductive ZSL Framework

Logistic Regression classifier trained on seen classes

Predicted embedding vectors mapped to closest unseen class

Evaluation performed on unseen classes only

Results

Accuracy on unseen classes (Benign vs Malignant): 60%

Demonstrated generalization to unseen tumor types

Confusion matrix analysis performed

Tech Stack

Python

TensorFlow / Keras

Scikit-learn

Gensim (Word2Vec)

NumPy / Pandas

Key Learnings

Embedding alignment between visual and semantic spaces

Transfer learning for feature extraction

Zero-shot inference pipeline

Handling computational constraints in deep learning

Future Improvements

Replace logistic regression with deep metric learning

Use contextual embeddings (BERT-based)

Deploy inference API using FastAPI
