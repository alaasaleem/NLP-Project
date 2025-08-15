# src/preprocess.py
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import ISRIStemmer
from collections import defaultdict

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def load_data(excel_path, csv_path):
    try:
        df = pd.read_excel(excel_path)
    except FileNotFoundError:
        df = pd.read_csv(csv_path)
    return df

def tokenize_arabic_text(text):
    if pd.isna(text) or text is None:
        return []
    
    text = str(text)
    
    return text.split()

def remove_stopwords(tokens):
    if not tokens:  
        return []
    
    try:
        arabic_stopwords = set(stopwords.words('arabic'))
    except LookupError:
        arabic_stopwords = {'في', 'من', 'إلى', 'على', 'هذا', 'هذه', 'ذلك', 'التي', 'الذي', 'أن', 'كان', 'كانت'}
    
    return [token for token in tokens if token not in arabic_stopwords]

def stem_tokens(tokens):
    if not tokens:  
        return []
    
    stemmer = ISRIStemmer()
    return [stemmer.stem(token) for token in tokens]

def preprocess_documents(df):
    df = df.dropna(subset=['Document'])
    
    df = df.reset_index(drop=True)
    
    df['Document'] = df['Document'].astype(str)
    
    return df

def create_inverted_index(df):
    inverted_index = defaultdict(list)
    for index, row in df.iterrows():
        doc_id = row['QID']
        tokens = row['Tokenized_Document']
        
        if isinstance(tokens, str):
            tokens = tokens.split()
        
        for token in tokens:
            if doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)
    return inverted_index