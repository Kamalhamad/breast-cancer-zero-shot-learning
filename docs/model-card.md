# Model Card — Breast Cancer ZSL Baseline

## Model details
- **Type:** Inductive Zero-Shot Learning baseline
- **Visual encoder:** ResNet50 (ImageNet pretrained)
- **Semantic encoder:** Word2Vec class-label embeddings
- **Classifier:** Logistic Regression on seen classes

## Intended use
- Educational/research demonstration of ZSL in medical imaging workflows.
- Portfolio artifact illustrating reproducibility and engineering hygiene.

## Not intended use
- Clinical decision support.
- Autonomous diagnosis or treatment recommendations.

## Data
- BreakHis histopathology dataset (as described in project docs).

## Metrics
- Primary tracked metrics:
  - train accuracy
  - inference accuracy
  - confusion matrix
  - classification report

## Limitations and risks
- Synthetic mode is for scaffolding only and not representative of clinical performance.
- Dataset shifts and class imbalance can significantly change results.
- Embedding quality for class labels can constrain zero-shot transfer performance.
