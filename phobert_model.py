import torch
from transformers import RobertaForSequenceClassification, AutoTokenizer
from typing import List, Tuple
import streamlit as st


MODEL_REPO = "Steven2002/Finetune-PhoBERT-15KComment"


@st.cache_resource
def load_model():
    """Load PhoBERT model and tokenizer from Hugging Face Hub"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)
        model = RobertaForSequenceClassification.from_pretrained(MODEL_REPO)
        return tokenizer, model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info(f"Please check if the model '{MODEL_REPO}' exists and is accessible.")
        return None, None

def predict_sentiment(text: str, tokenizer, model) -> Tuple[str, float]:
    """Predict sentiment for a single text"""
    if not text.strip():
        return "neu", 0.0
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probabilities = torch.softmax(outputs.logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
    label_mapping = {0: "neg", 1: "neu", 2: "pos"}
    predicted_label = label_mapping.get(predicted_class, "neu")
    return predicted_label, confidence

def predict_batch(texts: List[str], tokenizer, model, batch_size: int = 32) -> List[Tuple[str, float]]:
    """Predict sentiment for a batch of texts"""
    if not texts:
        return []
    results = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts, 
            return_tensors="pt", 
            truncation=True, 
            max_length=512, 
            padding=True
        )
        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
        label_mapping = {0: "neg", 1: "neu", 2: "pos"}
        for pred_class, confidence in zip(predicted_classes, confidences):
            predicted_label = label_mapping.get(pred_class.item(), "neu")
            results.append((predicted_label, confidence.item()))
    return results 