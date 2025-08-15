# src/search.py
from src.preprocess import tokenize_arabic_text, remove_stopwords, stem_tokens
from nltk.stem import ISRIStemmer

def search(term, inverted_index):
    stemmer = ISRIStemmer()
    term = stemmer.stem(term.lower())
    return inverted_index.get(term, [])

def preprocess_query(query):
    question_words = ['ماذا', 'من', 'أين', 'متى', 'كيف', 'لماذا', 'كم', 'أي', 'هل']
    
    tokens = tokenize_arabic_text(query)
    tokens = [tok for tok in tokens if tok not in question_words]
    tokens = remove_stopwords(tokens)
    tokens = stem_tokens(tokens)
    return tokens
