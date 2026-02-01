# 📩 SMS Spam Detection - NLP Project

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://spam-detection-nlp.streamlit.app/)

A robust Machine Learning powered SMS Spam detection system that classifies messages as **Spam** or **Ham** (Not Spam) with high accuracy. This project explores various NLP techniques and machine learning models to provide reliable predictions.

## 🚀 Live Demo
Try the model yourself here: [Spam Detection Web App](https://spam-detection-nlp.streamlit.app/)

## ✨ Key Features
- **Real-time Prediction**: Instantly classify any SMS or text message.
- **Confidence Scoring**: View the model's confidence for each prediction.
- **Advanced NLP Pipeline**: Implements tokenization, lemmatization, and stop-word removal.
- **Multi-Model Analysis**: Evaluates Naive Bayes, Logistic Regression, and Word Embeddings (FastText).

## 🛠️ Tech Stack
- **Data Processing**: [Polars](https://pola.rs/) (High-performance alternative to Pandas)
- **Natural Language Processing**: [NLTK](https://www.nltk.org/)
- **Machine Learning**: [Scikit-learn](https://scikit-learn.org/)
- **Vectorization**: Bag-of-Words (BoW), TF-IDF, FastText Embeddings
- **Web Framework**: [Streamlit](https://streamlit.io/)

## 📊 Performance Summary
During development, several models were evaluated. The **Multinomial Naive Bayes (BoW)** model emerged as the most effective for this dataset.

| Model | Feature Engineering | Accuracy |
| :--- | :--- | :--- |
| **Naive Bayes** | **Bag-of-Words** | **~97.5%** |
| Logistic Regression | Bag-of-Words | ~97.2% |
| Naive Bayes | TF-IDF | ~95.6% |
| Logistic Regression | FastText (Dense) | ~93.0% |

## 📁 Project Structure
```text
├── Data/                   # Raw dataset (spam.csv)
├── Notebook/               # Data exploration & model training
│   └── spamDetection.ipynb # Jupyter Notebook with full pipeline
├── app.py                  # Streamlit web application
├── bow_vectorizer.pkl      # Saved BoW vectorizer
├── spam_bow_nb.pkl         # Saved Naive Bayes model
└── requirements.txt        # Project dependencies
```

## ⚙️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Hannanbutt1256/SpamDetection_NLP.git
   cd SpamDetection_NLP
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

---
*Created by [Hannan Butt](https://github.com/Hannanbutt1256)*