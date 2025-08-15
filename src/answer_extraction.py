# src/answer_extraction.py
from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoModelForQuestionAnswering, pipeline

# NER
def load_ner_model(model_name="asafaya/bert-base-arabic"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    return pipeline("ner", model=model, tokenizer=tokenizer)

def identify_entities(text, ner_pipeline):
    return ner_pipeline(text)

# QA
def load_qa_model(model_name="bert-base-multilingual-cased"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return pipeline("question-answering", model=model, tokenizer=tokenizer)

def extract_precise_answer(document, question, qa_pipeline):
    result = qa_pipeline(question=question, context=document)
    return result['answer']
