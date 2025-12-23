# NLP & LLM Evaluation Pipeline  
**Prompt Engineering, Classical NLP Baselines, and Lightweight LLM Fine-Tuning**

## Overview
This repository implements an end-to-end **NLP experimentation and evaluation pipeline** focused on comparing classical supervised models with prompt-based and parameter-efficient LLM approaches. The project emphasizes **rigorous evaluation, reproducibility, and engineering discipline**, aligned with real-world ML workflows.

The core task is **sentiment analysis on the IMDb movie reviews dataset**, extended with prompt engineering, retrieval-style search, and deployment-ready APIs.

---

## Key Objectives
- Build **strong NLP baselines** (Logistic Regression, SVM)
- Design and evaluate **prompt-engineering strategies** (0-shot, 1-shot, multi-shot)
- Fine-tune LLMs using **LoRA** under compute constraints
- Perform **robust evaluation** (accuracy, latency, ranking metrics)
- Package models for **API and Streamlit deployment**
- Ensure **reproducibility and experiment tracking**

---

## Project Structure

```

week4-nlp-llms/
│
├── api/                    # FastAPI routes for inference
├── data/
│   ├── raw/                # (ignored) original datasets
│   ├── processed/          # (ignored) cleaned data
│   ├── splits/             # train/val/test splits
│   └── stress_tests/       # adversarial / robustness tests
│
├── docs/                   # Reports and notes
├── experiments/
│   └── configs/            # Experiment configurations
│
├── notebooks/              # EDA, prompting, evaluation notebooks
├── outputs/                # (ignored) models, logs, figures
├── prompts/                # Prompt templates & variants
├── scripts/                # Training and evaluation scripts
├── search_engine/          # Lightweight semantic search pipeline
├── src/
│   ├── data/               # Data loading & validation
│   ├── models/             # Classical & LLM-based models
│   ├── evaluation/         # Metrics and benchmarks
│   └── search/             # Indexing & retrieval logic
│
├── streamlit_app/          # Interactive demo UI
├── tests/                  # Unit tests (data, metrics, pipelines)
├── wandb/                  # (ignored) experiment tracking
└── README.md

````

---

## Models & Methods

### Classical NLP Baselines
- TF-IDF + Logistic Regression  
- TF-IDF + Linear SVM  
- Strong baselines used for comparison and sanity checks

### Prompt Engineering
- ≥ 5 prompt templates (0-shot / 1-shot / 3-shot)
- Logged:
  - Prompt text
  - Model response
  - Latency
  - Accuracy on held-out set

### LLM Fine-Tuning
- Parameter-Efficient Fine-Tuning (LoRA)
- Checkpointed training
- Compared against prompting and classical baselines

---

## Evaluation
- Accuracy, Precision, Recall, F1
- Ranking metrics (MRR, nDCG) for search tasks
- Latency benchmarking
- Stress tests for robustness
- Reproducible runs with fixed seeds

---

## Experiment Tracking & Reproducibility
- **Weights & Biases** for experiment logging
- Fixed random seeds
- Structured configs
- Clear separation of code, data, and artifacts

---

## Deployment
- **FastAPI** for model inference
- **Streamlit** application for interactive exploration
- Serialized models for reuse

---

## How to Run

### 1. Environment Setup
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
````

### 2. Data

Download the IMDb dataset and place it under:

```
data/raw/aclImdb/
```

### 3. Train Baselines

```bash
python scripts/train_baseline.py
```

### 4. Run Evaluation

```bash
python scripts/evaluate.py
```

### 5. Launch API

```bash
uvicorn api.main:app --reload
```

### 6. Streamlit App

```bash
streamlit run streamlit_app/app.py
```

---

## Key Takeaways

* Prompt engineering can be competitive with classical ML for certain tasks
* Strong baselines remain essential for honest evaluation
* LoRA enables efficient LLM fine-tuning under resource constraints
* Reproducibility and logging are as important as model accuracy

---

## Future Work

* Larger-scale retrieval-augmented generation (RAG)
* Better calibration and uncertainty estimation
* Dataset shift and robustness benchmarking
* Automated prompt search
