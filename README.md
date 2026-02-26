# Breast Cancer Classification using Inductive Zero-Shot Learning

## Overview
Research/portfolio project implementing an Inductive Zero-Shot Learning (ZSL) pipeline on the BreakHis breast cancer histopathology dataset.

> Not for clinical use.

## Method (high level)
- Visual features: pretrained ResNet50 feature extraction
- Semantic embeddings: Word2Vec embeddings for class labels
- Classifier: Logistic Regression trained on seen classes
- ZSL inference: similarity-based mapping for unseen classes

## Project Structure
`	ext
src/
└── bczsl/
    ├── __init__.py
    ├── __main__.py
    ├── data_processing.py
    ├── feature_extraction.py
    ├── embeddings.py
    ├── train.py
    ├── zsl_inference.py
    └── evaluation.py


 = @"

 =@"




>> `
>>
>> ## Quickstart (dev install)
>> `ash
>> python -m venv .venv
>> .\.venv\Scripts\Activate.ps1
>>
>> pip install -r requirements.txt
>> pip install -e .
>> python -m bczsl
>> `
>>
>> Roadmap: see docs/PORTFOLIO_UPGRADE_PLAN.md
