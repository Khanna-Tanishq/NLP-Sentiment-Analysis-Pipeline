# NLP-Sentiment-Analysis-Pipeline
This project builds an end-to-end Mathematical Linguistics pipeline for sentiment analysis. It processes unstructured text using negation-preserving tokenization, POS-guided lemmatization, and TF-IDF vectorization in sparse CSR format, predicting sentiment with a Naive Bayes classifier reaching 85.50% accuracy on IMDb data.


# NLP & Sentiment Analysis Engine: Mathematical Linguistics Pipeline

An end-to-end Natural Language Processing (NLP) engine designed to transform unstructured human text into sparse mathematical representations and accurately classify emotional sentiment polarity using probabilistic inference.

---

## 📌 Project Overview

In enterprise data engineering, over **80% of data exists in unstructured, qualitative formats** (product reviews, emails, support tickets)[cite: 1]. Traditional analytics tools cannot process raw human language directly because machine learning algorithms require mathematical structures[cite: 1].

This project implements a **Mathematical Linguistics Pipeline** built to solve the unstructured text problem[cite: 1]. It handles raw text ingestion, character normalization, negation-aware stop-word filtering, POS-guided lemmatization, TF-IDF sparse matrix vectorization, and probabilistic classification using Multinomial Naive Bayes[cite: 1].

---

## 📊 Performance & Benchmark Metrics

The engine was evaluated on the benchmark **IMDb Movie Reviews Dataset**, achieving strong predictive performance:

* **Model Accuracy**: **`85.50%`**
* **Target Classes**: Binary (`0` = Negative, `1` = Positive)
* **Classifier**: Multinomial Naive Bayes with Laplace Smoothing ($\alpha = 1.0$)[cite: 1]

---

## 🛠️ Architecture & How It Works

The system bridges unstructured text and quantitative spatial representations through a strict 5-stage assembly line:

```text
┌──────────┐     ┌──────────────┐     ┌─────────────────┐     ┌──────────────────────┐
│ 1. INGEST│ ──► │ 2.PRE-PROCESS│ ──► │  3. VECTORIZE   │ ──► │ 4. PROBABILISTIC MODEL│
│Raw String│     │Clean Tokens  │     │Sparse CSR Matrix│     │ Positive / Negative  │
└──────────┘     └──────────────┘     └─────────────────┘     └──────────────────────┘

```

### Key Engineering Rules Implemented

1. **Negation Retention in Stop-Words**: Default NLP stop-word dictionaries strip out crucial negation terms (`"not"`, `"never"`, `"no"`), leading to severe sentiment inversion (e.g., converting *"not happy"* into *"happy"*). Our preprocessor explicitly preserves negations using set-based operations.


2. **POS-Guided Lemmatization**: Standard stemming uses crude heuristics that truncate words into non-dictionary stems (e.g., `"studying"` → `"studi"`). We extract Part-of-Speech (POS) tags from Treebank and map them to WordNet to ensure proper morphological reduction (e.g., `"went"` → `"go"`).


3. **N-Gram TF-IDF Vectorization**: We use Term Frequency-Inverse Document Frequency (TF-IDF) with both unigrams and bigrams (`ngram_range=(1,2)`) to capture contextual modifier pairs like `"not good"`.


4. **Compressed Sparse Row (CSR) Optimization**: Dense high-dimensional matrices exhaust RAM rapidly ($O(N^3)$ operations). Storing vectorized text strictly in SciPy CSR format ignores zero values, saving gigabytes of memory.


5. **Probabilistic Naive Bayes with Laplace Smoothing**: To fix the zero-frequency problem where an unseen testing word zeroes out the entire posterior probability, we apply Laplace smoothing ($\alpha = 1.0$) as a pseudocount safeguard.



---

## 💻 Tech Stack & Dependencies

* **Language:** Python 3.8+
* **NLP & Text Processing:** NLTK (`wordnet`, `punkt`, `averaged_perceptron_tagger`, `stopwords`)
* **Vectorization & Modeling:** Scikit-Learn (`TfidfVectorizer`, `MultinomialNB`, `ComplementNB`)
* **Matrix Computation:** SciPy (CSR Sparse Matrices), NumPy, Pandas
* **Visualization:** Matplotlib, Seaborn

---

## ⚙️ Installation & Usage Guide

### 1. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)<your-username>/NLP-Sentiment-Analysis-Pipeline.git
cd NLP-Sentiment-Analysis-Pipeline

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

### 3. Run the Pipeline

Open `NLP_Sentiment_Analysis_Pipeline.ipynb` in **Google Colab** or **Jupyter Notebook** and select **Run All**.

---

## 🔍 Inference Example

```python
from NLP_Sentiment_Analysis_Pipeline import TextPreprocessor, tfidf, model

processor = TextPreprocessor()

# Raw text evaluation
input_review = "The product is not good and stopped working after two days!"

# 1. Clean & Lemmatize (preserving negations)
cleaned_review = processor.clean_text(input_review)

# 2. Transform into TF-IDF Vector
vectorized_review = tfidf.transform([cleaned_review])

# 3. Classify Class
prediction = model.predict(vectorized_review)[0]
sentiment = "POSITIVE" if prediction == 1 else "NEGATIVE"

print(f"Processed Text : {cleaned_review}")
print(f"Predicted Class: {sentiment}")
# Output: Processed Text : not good stop work two day
# Output: Predicted Class: NEGATIVE

```

---

## 📂 Repository Structure

```text
NLP-Sentiment-Analysis-Pipeline/
│
├── data/
│   └── IMDB Dataset.csv                     # Raw reviews dataset
│
├── NLP_Sentiment_Analysis_Pipeline.ipynb     # Master Google Colab Notebook
├── README.md                                 # Project documentation & benchmark analysis
└── requirements.txt                          # Dependencies list

```

---

## 🎓 Credit & Acknowledgments

* **Batch:** 2026
* **Program:** Industrial Training Kit — Powered by **DecodeLabs**

```

```
