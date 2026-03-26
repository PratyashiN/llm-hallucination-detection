# LLM Hallucination Detection using Internal Confidence Signals

##  Overview

Large Language Models (LLMs) such as LLaMA often generate fluent and confident responses, but they can also produce **hallucinations** — answers that are factually incorrect or misleading.

This project investigates whether hallucinations can be detected using **internal confidence signals** (like entropy and token probabilities) without relying on external fact-checking systems.

---

##  Problem Statement

LLMs do not explicitly indicate uncertainty and may generate incorrect answers with high confidence.

The key question explored in this project is:

> Can hallucinations in LLM outputs be detected using only internal model signals?

---

##  Methodology

The project follows a structured pipeline:

### 1. Dataset
- Used **TruthfulQA**, a benchmark designed to evaluate truthfulness in LLM responses.
- Contains:
  - Questions
  - Correct answers
  - Common misconceptions (incorrect answers)

---

### 2. Model
- Used **LLaMA-2 (7B Chat)** via Hugging Face Transformers.
- Generated responses for each question in the dataset.

---

### 3. Signal Extraction (Core Contribution)

For each generated response, internal signals were extracted:

- **Token Log Probabilities**
- **Entropy (uncertainty measure)**

These signals reflect the model’s confidence during generation.

---

### 4. Feature Engineering

Token-level signals were aggregated into meaningful features:

- Average, minimum, and standard deviation of log probabilities  
- Average, maximum, and standard deviation of entropy  
- Response length  
- Number of low-confidence tokens  
- Number of high-entropy tokens  

---

### 5. Labeling Strategy

Each response was labeled as:

- **1 (Correct)** → if it matches ground truth answers  
- **0 (Hallucination)** → otherwise  

A hybrid approach was used:
- Exact match
- Keyword overlap threshold

---

### 6. Classification Model

- Trained a **Logistic Regression classifier** with class balancing
- Input: Extracted features  
- Output: Hallucination vs Correct prediction  

---

### 7. Evaluation

- Used train-test split with stratification
- Evaluated using:
  - Precision
  - Recall
  - F1-score

---

##  Results & Observations

- The classifier showed **limited performance** in detecting hallucinations
- Significant class imbalance was observed in dataset outputs
- Even with improved features, performance remained inconsistent

---

##  Key Insight

> LLMs can produce **high-confidence incorrect answers**, making hallucination detection using only internal uncertainty signals inherently challenging.

This highlights a critical limitation:

- **Low entropy ≠ correctness**
- **High confidence does not guarantee truthfulness**

---

## Learning Outcomes

Through this project, the following key concepts were explored:

- Internal mechanics of LLM generation  
- Token-level probability analysis  
- Entropy as a measure of uncertainty  
- Feature engineering for NLP  
- Handling class imbalance in ML  
- Model evaluation pitfalls (overfitting, bias)

---

##  Tech Stack

- Python  
- Hugging Face Transformers  
- PyTorch  
- Scikit-learn  
- Pandas, NumPy  
- Google Colab  

---


---

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2. Open the notebook
3. Run all cells sequentially

##  Conclusion

This project demonstrates that while internal confidence signals provide useful insights, they are not sufficient on their own to reliably detect hallucinations in LLMs.

A hybrid approach combining internal signals + external verification is likely required for robust systems.
