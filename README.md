# Arabic Question Answering System (QA) 🔥

![Arabic QA](https://img.shields.io/badge/Language-Arabic-blue?style=flat-square) ![Python](https://img.shields.io/badge/Tech-Python-yellow?style=flat-square) ![NLP](https://img.shields.io/badge/Tech-NLP%2C%20BERT-orange?style=flat-square)

## 🚀 Project Overview
This project is an **Arabic Question Answering (QA) system** that allows users to ask questions in Arabic and receive **accurate answers** from a large document corpus.  
It combines **information retrieval techniques** with **state-of-the-art NLP models** for Arabic.

## 🛠 What We Did
- Collected and preprocessed **Arabic documents** (tokenization, stopword removal, stemming).  
- Built an **inverted index** for fast document retrieval.  
- Vectorized documents using **TF-IDF** for similarity search.  
- Implemented a **search module** using cosine similarity to find relevant documents.  
- Applied **BERT-based NER** and **QA pipelines** to extract answers from top-ranked documents.  
- Evaluated the system using **precision, recall, and F1-score**.

## 📊 Key Results
- Retrieved **relevant documents** accurately.  
- Extracted **correct answers** from top documents.  
- Achieved **F1-score = 0.71**, **Recall = 1.0**, **Precision = 0.56**.

## ⚡ Tech Stack
- Python 
- Pandas, Numpy  
- Scikit-learn (TF-IDF, Cosine Similarity)  
- NLTK (Arabic stemming)  
- Hugging Face Transformers (BERT for Arabic NER & QA)  
