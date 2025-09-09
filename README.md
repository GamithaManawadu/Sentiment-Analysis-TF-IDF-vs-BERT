# 😊 Sentiment Analysis: TF-IDF vs BERT

A comprehensive comparison of **three NLP approaches** to sentiment classification on IMDB movie reviews using: traditional ML, zero-shot transfer learning(BERT), and fine-tuned transformers.

> **💡 [View the Interactive Explanation →](https://htmlpreview.github.io/?https://github.com/GamithaManawadu/Sentiment-Analysis-TF-IDF-vs-BERT/blob/main/Explanations/sentiment-analysis-explained.html)**

---

## 🎯 Results

| Model                        | Approach          | Training on IMDB? | Accuracy  |
| ---------------------------- | ----------------- | :---------------: | :-------: |
| TF-IDF + Random Forest       | Traditional ML    |     Yes (25K)     |   85.1%   |
| TF-IDF + XGBoost             | Traditional ML    |     Yes (25K)     |   85.3%   |
| TF-IDF + Linear SVM          | Traditional ML    |     Yes (25K)     |   88.5%   |
| TF-IDF + Logistic Regression | Traditional ML    |     Yes (25K)     |   89.5%   |
| DistilBERT (Zero-Shot)       | Transfer Learning |      **NO**       |   91.7%   |
| DistilBERT (Fine-Tuned)      | Transfer Learning |     Yes (25K)     | **93.3%** |

**Key Finding:** Zero-shot BERT (91.7%) beats the best TF-IDF model (89.5%) without seeing a single IMDB training example. Fine-tuning pushes it to 93.3%.

---

## 📊 What's Inside

### Core Comparison

- **Approach A —> TF-IDF + Traditional ML:** 15,000 TF-IDF features (unigrams + bigrams), 4 models compared with cross-validation
- **Approach B —> BERT Zero-Shot:** Pre-trained DistilBERT (SST-2) applied directly - no IMDB training
- **Approach C —> BERT Fine-Tuned:** DistilBERT fine-tuned for 3 epochs on IMDB (LR=2e-5, batch=16)

### Advanced Analysis

- **Learning Curve:** TF-IDF accuracy from 100 → 25K training examples vs BERT's zero-shot baseline
- **RoBERTa Comparison:** 95.4% zero-shot on 500 reviews (vs DistilBERT 90.6%, TF-IDF 88.4%)
- **Ensemble (Soft Voting):** TF-IDF + BERT averaged → 92.4% (beats both individual models)
- **Speed Benchmark:** TF-IDF is 1,573× faster than BERT on CPU (0.3ms vs 498ms per review)
- **Error Analysis:** 370 cases where BERT wins vs 223 where TF-IDF wins - BERT handles sarcasm and nuance better
- **LIME Explainability:** Visual word-level explanations of BERT's black-box predictions
- **Negation-Aware Preprocessing:** Merging "not good" → "not_good" (+0.03% - minimal since bigrams already capture this)
- **VADER Features:** Adding rule-based sentiment scores alongside TF-IDF (-0.53% - actually hurt performance)
- **Domain Generalization:** Both models score 100% on Amazon-style product reviews
- **Gradio Interactive Demo:** Web UI for live predictions from both models

---

## 🗂️ Repository Structure

```
├── sentiment-analysis.ipynb              # Main notebook (all code + analysis)
├── requirements.txt                      # Dependencies
├── Explanations/
│   └── sentiment-analysis-explained.html # Interactive visual explanation
└── README.md
```

---

## 💡 Key Takeaways

- **Transfer learning wins in NLP** - same pattern as computer vision (MobileNetV2 crushing from-scratch CNNs)
- **Linear models beat tree-based models on sparse text** - LogReg > XGBoost on TF-IDF features (opposite of tabular data)
- **Speed vs accuracy is a real tradeoff** - TF-IDF at 1,573× faster may be the right choice for production at scale
- **Ensembling different model types works** - BERT + TF-IDF soft vote (92.4%) beats either model alone

---

## ⚙️ Setup

```bash
git clone https://github.com/GamithaManawadu/Sentiment-Analysis-TF-IDF-vs-BERT.git
cd Sentiment-Analysis-TF-IDF-vs-BERT

python -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows

pip install -r requirements.txt
jupyter notebook sentiment-analysis.ipynb
```

---
