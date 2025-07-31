import streamlit as st
import pandas as pd
from phobert_model import load_model, predict_sentiment, predict_batch

def create_sample_csv():
    sample_data = [
        "Tôi rất thích sản phẩm này!",
        "Dịch vụ khách hàng không tốt.",
        "Sản phẩm bình thường, không có gì đặc biệt.",
        "Chất lượng sản phẩm rất tốt.",
        "Giao hàng chậm và sản phẩm bị hỏng.",
        "Tôi hài lòng với trải nghiệm mua sắm.",
        "Giá cả hợp lý và chất lượng tốt.",
        "Không nên mua sản phẩm này.",
        "Tuyệt vời! Tôi sẽ mua lại.",
        "Thất vọng với dịch vụ."
    ]
    df = pd.DataFrame({"text": sample_data})
    return df.to_csv(index=False)

def calculate_statistics(df):
    stats = {}
    label_counts = df['predicted_label'].value_counts()
    stats['label_counts'] = label_counts.to_dict()
    df['word_count'] = df['text'].str.split().str.len()
    stats['avg_word_count'] = df['word_count'].mean()
    df['char_count'] = df['text'].str.len()
    stats['avg_char_count'] = df['char_count'].mean()
    return stats

def inject_custom_css(theme):
    try:
        with open("custom_style.css") as f:
            css = f.read()
        theme_class = "light" if theme == "light" else "dark"
        st.markdown(f"<style>body, .stApp {{}} </style>", unsafe_allow_html=True)
        st.markdown(f'<style>body, .stApp {{}} </style>', unsafe_allow_html=True)
        st.markdown(f'<script>document.body.classList.add("{theme_class}"); document.querySelector(".stApp").classList.add("{theme_class}");</script>', unsafe_allow_html=True)
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Could not load custom CSS: {e}")

# Model loading and wrappers
@st.cache_resource
def get_model():
    return load_model()

def predict_sentiment_wrapper(text):
    tokenizer, model = get_model()
    return predict_sentiment(text, tokenizer, model)

def predict_batch_wrapper(texts):
    tokenizer, model = get_model()
    return predict_batch(texts, tokenizer, model, batch_size=32) 