# LegalTech – Court-Aware Mixture-of-Experts System for Legal Document Intelligence

## 1. Project Description

LegalTech is a court-aware legal document intelligence platform designed to analyze, route, and classify Indian legal case documents using a Mixture-of-Experts (MoE) deep learning architecture.

The system combines:
- Domain-adapted language models (Legal-BERT)
- Parameter-efficient fine-tuning (LoRA)
- Supervised neural routing
- Court-specialized expert classifiers
- A full-stack MERN-based deployment layer

The goal is to move beyond monolithic legal NLP models and instead enable **court-specific reasoning**, **scalable inference**, and **interpretability-driven analytics**.

---

## 2. Problem Motivation

Indian legal documents vary significantly across court hierarchies:
- Constitutional reasoning in Supreme Court cases
- Precedent-heavy arguments in High Courts
- Fact-dense procedural language in District Courts
- Regulatory tone in Tribunal orders
- Summary-style Daily Orders

A single model fails to capture this diversity effectively.  
LegalTech addresses this through **explicit expert specialization guided by a learned router**.

---

## 3. End-to-End Architecture

<img width="1777" height="744" alt="image" src="https://github.com/user-attachments/assets/23af228b-40e3-4880-9105-cb22c7cafd6f" />



The system follows a strictly modular pipeline:

### 3.1 LoRA Conversion
- Base encoder: Legal-BERT
- LoRA adapters injected into attention and feed-forward layers
- Enables memory-efficient fine-tuning without full model retraining

### 3.2 Tokenization Layer
- Custom tokenizer trained on Indian legal corpora
- Preserves legal citations, section references, and court-specific terminology
- Fixed vocabulary for reproducible encoding

### 3.3 Encoding Pipeline
- Long documents split into overlapping chunks
- Each chunk encoded independently
- Mean pooling applied to generate document-level embeddings
- Output dimension: 768

### 3.4 Router Network
- Supervised multi-class classifier
- Input: document embeddings (+ optional metadata)
- Architecture: SE-Residual MLP
- Uses temperature annealing to transition from soft to hard routing
- Outputs a single court expert selection at inference

### 3.5 Court-Specific Experts
Each expert is trained independently:
- Supreme Court Expert
- High Court Expert
- District Court Expert
- Tribunal Expert
- Daily Order Expert

Experts learn court-specific linguistic and semantic distributions, improving accuracy and robustness.

### 3.6 Inference Flow
1. Document → Tokenization
2. Tokenized chunks → Encoder
3. Encoded representation → Router
4. Router selects expert
5. Expert produces final prediction
6. Metrics logged and visualized via frontend

---

## 4. Technology Stack

### Machine Learning
- Python
- PyTorch
- HuggingFace Transformers
- Legal-BERT
- LoRA (PEFT)

### Backend
- Node.js
- Express.js
- REST-based inference APIs

### Frontend
- React (Vite)
- Tailwind CSS
- Interactive dashboards and analytics

### Database
- MongoDB
- Stores users, logs, metrics, and metadata

### Infrastructure
- GPU-based training (A100 / DGX class systems)
- Mixed Precision Training (AMP)
- Checkpoint-based recovery

---

## 5. Training Strategy

- Stratified K-Fold Cross Validation
- Mixed Precision Training (AMP)
- Stochastic Weight Averaging (SWA)
- Exponential Moving Average (EMA)
- MixUp Regularization
- Asymmetric Focal Loss
- Cosine Learning Rate Scheduling

Each expert is optimized independently to avoid negative transfer.

---

## 6. Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Court-wise Confusion Matrices
- Router Entropy and Routing Confidence

Metrics are logged and visualized through the admin dashboard.

---

## 7. Deployment Overview

- Backend APIs handle encoding, routing, and inference
- Frontend dashboards display:
  - Court-wise predictions
  - Performance metrics
  - Confusion matrices
- System designed for modular scaling and expert expansion

---

## 9. Notes

- Large model files are excluded from version control
- GPU is recommended for training and batch inference
  
---

## 11. License

This project is intended for academic and research purposes.  
All rights reserved by the authors.
