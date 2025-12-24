# LegalTech – Court-Aware AI System for Legal Document Analysis

## Overview

LegalTech is an end-to-end AI-driven legal document analysis platform designed to process, classify, and route Indian legal case documents across different court types.  
The system leverages modern deep learning architectures, domain-adapted language models, and a Mixture-of-Experts (MoE) framework to achieve court-specific specialization and improved prediction accuracy.

This repository contains the complete backend ML pipeline along with the web application stack used for deployment and visualization.

---

## Key Objectives

- Automate legal document understanding using domain-specific language models
- Enable court-aware routing using a supervised router network
- Train specialized expert models for different court categories
- Provide an interactive web interface for analysis, metrics, and inference
- Maintain scalability, modularity, and reproducibility

---

## Architecture Summary

The system follows a modular, production-oriented pipeline:

1. **LoRA Conversion**
   - Conversion of the dataset into a structured format , tailored to model understanding.

2. **Tokenization**
   - Customized tokenizer (Base - LegalBERT) using chunking methodology

3. **Encoding**
   - Chunk-based embedding generation with pooling strategies

4. **Router Network**
   - Supervised MLP-based router with temperature annealing
   - Predicts the most relevant court expert for each document

5. **Expert Models**
   - Independent court-specific classifiers
   - Trained using advanced optimization strategies

6. **Inference & Deployment**
   - Integrated with a MERN-based web platform

---

## Court Experts Supported

- Supreme Court  
- High Court  
- District Court  
- Tribunal  
- Daily Orders  

Each expert is trained independently to capture court-specific linguistic and procedural patterns.

---

## Technology Stack

### Machine Learning & AI
- Python
- PyTorch
- HuggingFace Transformers
- Legal-BERT
- LoRA (Low-Rank Adaptation)

### Backend
- Node.js
- Express.js

### Frontend
- React (Vite)
- Tailwind CSS

### Database
- MongoDB

### Infrastructure
- GPU-based training (DGX / A100)
- Model checkpoints stored as `.pt` and `.pth` files

---

## Repository Structure

Court - MOE/
│
├── encoding/ # Pre-encoded embeddings
├── tokenizer/ # Custom tokenizer files
├── lora/ # LoRA adapters and configs
├── router/ # Court routing model
├── experts/ # Court-specific expert models
├── inference/ # Inference and evaluation scripts
├── backend/ # Node + Express backend
├── frontend/ # React + Tailwind frontend
├── metrics/ # Confusion matrices and evaluation
├── docs/ # Documentation and reports
└── README.md

---

## Training Methodologies

- Stratified K-Fold Cross Validation
- Mixed Precision Training (AMP)
- Stochastic Weight Averaging (SWA)
- Exponential Moving Average (EMA)
- MixUp Regularization
- Asymmetric Focal Loss
- Cosine Learning Rate Scheduling

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrices (per expert)
- Router Entropy and Routing Confidence

---

## Deployment

The system is deployed as a full-stack web application with:

- Backend APIs for inference and analytics
- Frontend dashboards for court-wise insights
- Secure model loading and inference routing

---
 
## License

This project is intended for academic, research, and educational purposes.  
All rights reserved by the authors.

---

## Notes

- Large model files are excluded from version control.
- Ensure GPU availability before running training scripts.
