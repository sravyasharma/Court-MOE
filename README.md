# ⚖️ Court-MOE

### LegalTech – AI Powered Legal Judgment Prediction

LegalTech is an advanced **AI-driven legal analytics system** designed to analyze Indian court case documents and predict judicial outcomes using modern **Natural Language Processing (NLP)** and **Mixture-of-Experts (MoE)** architectures.

The system processes complete legal judgments, extracts contextual reasoning patterns using **transformer-based encoders**, and predicts whether an appeal is **accepted or rejected**.

The model incorporates **court-specialized expert networks** to capture differences in legal reasoning across judicial hierarchies.

This project contributes toward **AI-assisted legal decision analysis**, helping reduce the analysis time for large-scale legal datasets.

---

# 📌 Project Description

**Court-MOE** is a **Mixture-of-Experts (MoE)** architecture developed to analyze and classify judgments from Indian courts.

The system captures **unique reasoning patterns across different levels of the judiciary**.

The model integrates multiple components to process legal documents:

* Custom tokenizer designed for legal text
* Fine-tuned **LegalBERT encoder** for contextual representations
* Lightweight **router network** that dynamically selects expert models
* **Five specialized expert networks**, each trained on a specific court category

---

# 🧠 Court Types and Expert Models

| Court Type            | Description                                                                  | Expert Model          |
| --------------------- | ---------------------------------------------------------------------------- | --------------------- |
| 🏛 **Supreme Court**  | Highest judicial authority handling constitutional matters and major appeals | Supreme Court Expert  |
| 🏛 **High Court**     | State-level courts hearing appeals from lower courts                         | High Court Expert     |
| 🏛 **District Court** | Handles majority of civil and criminal cases at the district level           | District Court Expert |
| ⚖ **Tribunals**       | Specialized courts dealing with domain-specific disputes                     | Tribunal Expert       |
| 📜 **Daily Orders**   | Procedural or interim rulings issued during case proceedings                 | Daily Orders Expert   |

Each expert is trained exclusively on **data from its respective court**, allowing the system to capture **domain-specific legal reasoning patterns**.

---

# 🎯 Why Court-MOE?

Legal judgments across courts follow **different writing styles, legal terminology, and reasoning structures**. A single monolithic model often struggles to learn these variations.

Court-MOE solves this using a **Mixture-of-Experts architecture**, where multiple specialized models focus on different court types.

Instead of forcing one model to learn all judicial patterns, the system **routes each case to the most relevant expert networks**.

### Advantages

* **Court-Specific Learning**
  Each expert focuses on patterns from its court level.

* **Better Generalization**
  The model adapts to differences between Supreme Court, High Court, District Court, and Tribunal decisions.

* **Efficient Modeling**
  Only relevant experts are activated rather than the entire model.

* **Improved Prediction Quality**
  Combining outputs from specialized experts leads to more accurate predictions.

Court-MOE therefore models the **diversity of judicial decision-making more effectively** than traditional single-model approaches.

---

# 🏗️ System Architecture

The model follows a **7-stage machine learning pipeline** designed for legal documents.

<p align="center">
<img width="900" alt="Court-MOE Architecture" src="https://github.com/user-attachments/assets/076147b0-9e32-4bc8-a64d-8b871396e079">
</p>

---

# 🔡 Custom Tokenizer

A **domain-trained BPE tokenizer** designed for Indian legal documents.

Handles:

* Legal citations
* Act names
* Latin legal terminology
* Multilingual text
* Procedural markers

---

# 🧩 LegalBERT Encoder

A fine-tuned **LegalBERT model** adapted for Indian legal data.

It generates **context-aware embeddings** that capture:

* Legal reasoning
* Case structure
* Semantic relationships between legal arguments

---

# 🧭 Router Network

A **4-layer SE-Residual MLP** that predicts the most relevant court expert.

Features:

* Metadata-augmented routing
* Dynamic expert selection
* Improved predictions for District Courts and Daily Orders

---

# 👩‍⚖️ Expert Models

Five expert networks — one for each court type.

Each expert is trained using advanced techniques:

| Technique                | Purpose                   |
| ------------------------ | ------------------------- |
| 3-Fold Stratified K-Fold | Robust validation         |
| AMP (Mixed Precision)    | Faster training           |
| SWA                      | Improved generalization   |
| EMA                      | Stability during training |
| MixUp                    | Better robustness         |
| Asymmetric Focal Loss    | Handles class imbalance   |

Each expert learns **court-specific writing styles, legal complexity, and reasoning patterns**.

---

# 🚀 Running Inference

Run the interactive CLI:

```bash
python prediction.py
```

You will be prompted for:

```
📂 Enter path to case file:
⚖️ Enter court type (press Enter to auto-detect):
```

### Two Modes

**Auto Mode (Recommended)**
Leave court type blank → Router selects the expert → Prediction generated.

**Manual Mode**

Specify a court type:

```
supreme
high
district
tribunal
daily
```

The system will **bypass routing and directly use the selected expert**.

---

# 📊 Performance Summary

## 🧭 Router

| Metric          | Value                          |
| --------------- | ------------------------------ |
| Accuracy        | ~62.7%                         |
| Macro F1        | ~0.75                          |
| Key Improvement | District Courts & Daily Orders |

---

## 👩‍⚖️ Experts

| Court Type     | Performance                          |
| -------------- | ------------------------------------ |
| Supreme Court  | Strong                               |
| High Court     | Strong                               |
| Tribunal       | Strong                               |
| District Court | Improved after metadata augmentation |
| Daily Orders   | Significant improvement              |

Each expert includes **confusion matrices and detailed evaluation reports**.

---

# 🎯 Vision & Roadmap

Court-MOE is built on the idea that:

> **Legal AI should respect the structural diversity of courts.**

Upcoming developments:

* 🐳 **Docker support** for easier deployment
* 🔍 **Explainability module** for token-level and section-level insights
* 📡 **REST API** for integration with legal platforms

---

# 📊 Dataset

The model is trained on **NyayaAnumana**, one of the largest datasets of Indian legal judgments.

### Dataset Characteristics

| Feature       | Value                      |
| ------------- | -------------------------- |
| Total Cases   | 700,000+                   |
| Document Type | Long-form legal judgments  |
| Labels        | Accept / Reject            |
| Coverage      | Multiple court hierarchies |

### Courts Included

* Supreme Court
* High Courts
* Tribunals
* District Courts
* Daily Orders

---

# 🛠️ Technologies Used

### Machine Learning

* Python
* PyTorch
* HuggingFace Transformers
* Legal-BERT
* LoRA (PEFT)

### Backend

* Node.js
* Express.js
* REST-based inference APIs

### Frontend

* React (Vite)
* Tailwind CSS
* Interactive dashboards and analytics

### Database

* MongoDB
* Stores users, logs, metrics, and metadata3


### Infrastructure

* GPU-based training (A100 / DGX class systems)
* Mixed Precision Training (AMP)
* Checkpoint-based recovery

---

# ⚙️ Implementation Details

Key techniques:

* Legal document chunking for long texts
* Domain-specific tokenization
* Dynamic expert routing
* Transformer-based contextual embeddings
* Weighted expert aggregation

Optimization methods:

* AdamW optimizer
* Binary cross-entropy loss
* Dropout regularization

---

# 📈 Applications

LegalTech can support several real-world legal applications:

### Legal Research Assistance

AI-assisted analysis of case outcomes.

### Judgment Prediction

Forecasting likely outcomes of legal appeals.

### Legal Document Understanding

Automated extraction of legal reasoning patterns.

### Judicial Analytics

Understanding trends across courts.

---

# 👨‍💻 Contributors

This project was developed by a **team of 5 researchers and engineers** working on AI applications in legal systems.

---

# 📜 License

This project is intended for **research and educational purposes**.

---

# ⭐ Acknowledgements

* Legal AI research community
* HuggingFace Transformers
* NyayaAnumana dataset contributors

---
