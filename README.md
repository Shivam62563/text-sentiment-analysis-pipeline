# Multi-Class Text Emotion Classification Pipeline

An end-to-end NLP project that classifies raw text into emotional states (joy, sadness, anger, fear) using traditional machine learning. Built to practice text preprocessing, feature engineering, and rigorous model evaluation.

## Tech Stack
* **Language:** Python, Pandas, NumPy
* **Text Processing:** NLTK (Tokenization, Stop Words Removal)
* **Modeling:** Scikit-Learn (CountVectorizer, TfidfVectorizer, MultinomialNB, Logistic Regression)

## Pipeline Overview
1. **Preprocessing:** Lowercasing, punctuation/number/emoji removal, and stop word filtering on 16k+ rows.
2. **Feature Engineering:** Text vectorization using Bag of Words and TF-IDF with strict train/test split isolation to prevent data leakage.
3. **Training & Evaluation:** Evaluated multiple classification algorithms and metrics.

## Model Performance
| Model | Feature Extraction | Test Accuracy |
| :--- | :--- | :--- |
| **Logistic Regression** | **TF-IDF Vectorizer** | **86.15%** |
| Naive Bayes | Bag of Words | 76.78% |
| Naive Bayes | TF-IDF Vectorizer | 66.09% |

## Quick Inference Demo
```python
sample_text = "I am feeling so thrilled and happy about this project!"
print(predict_emotion(sample_text))
