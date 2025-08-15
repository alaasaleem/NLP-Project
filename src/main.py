import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.preprocess import load_data, preprocess_documents, create_inverted_index, tokenize_arabic_text, remove_stopwords, stem_tokens
from src.search import preprocess_query, search
from src.answer_extraction import load_ner_model, identify_entities, load_qa_model, extract_precise_answer

df = load_data('data/QA.xlsx', 'data/QA.csv')
df = preprocess_documents(df)  

df['Tokenized_Document'] = df['Document'].apply(tokenize_arabic_text)
df['Tokenized_Document'] = df['Tokenized_Document'].apply(remove_stopwords)
df['Tokenized_Document'] = df['Tokenized_Document'].apply(stem_tokens)

inverted_index = create_inverted_index(df)

ner_pipeline = load_ner_model()
qa_pipeline = load_qa_model()

df['Tokenized_Document'] = df['Tokenized_Document'].apply(lambda tokens: ' '.join(tokens) if tokens else '')

tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(df['Tokenized_Document'])

def process_query(query, k=3):
    query_tokens = preprocess_query(query)
    query_vector = tfidf_vectorizer.transform([' '.join(query_tokens)])
    cosine_similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()
    related_docs_indices = cosine_similarities.argsort()[::-1]

    top_documents = []
    for index in related_docs_indices[:k]:
        doc_id = df.iloc[index]['QID'] 
        score = cosine_similarities[index]
        top_documents.append((doc_id, score))

    answers_with_ids = []
    for doc_id, _ in top_documents:
        doc_text = df[df['QID'] == doc_id]['Document'].values[0]
        
        if pd.notna(doc_text) and str(doc_text).strip():
            entities = identify_entities(str(doc_text), ner_pipeline)
            answer = extract_precise_answer(str(doc_text), query, qa_pipeline)
            answers_with_ids.append((answer, doc_id))
        else:
            answers_with_ids.append(("No valid document content found", doc_id))

    return answers_with_ids