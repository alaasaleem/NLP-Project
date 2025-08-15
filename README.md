# Arabic Question Answering System (QA) 

## 🚀 Project Overview
This project is an Arabic Question Answering (QA) system that allows users to ask questions in Arabic and receive accurate answers from a large document corpus.  
It combines information retrieval techniques with state-of-the-art NLP models for Arabic.

## 🛠 What We Did
- Collected and preprocessed Arabic documents (tokenization, stopword removal, stemming).  
- Built an inverted index for fast document retrieval.  
- Vectorized documents using TF-IDF for similarity search.  
- Implemented a search module** using cosine similarity to find relevant documents.  
- Applied BERT-based NER and QA pipelines to extract answers from top-ranked documents.  
- Evaluated the system using precision, recall, and F1-score.

## ⚡ Tech Stack
- Python 
- Pandas, Numpy  
- Scikit-learn (TF-IDF, Cosine Similarity)  
- NLTK (Arabic stemming)  
- Hugging Face Transformers (BERT for Arabic NER & QA)  
