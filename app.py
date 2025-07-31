import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import List, Tuple
import io
import base64
from phobert_model import load_model, predict_sentiment, predict_batch
from lang_dict import LANGS, _, get_lang, set_lang, get_theme, set_theme

# --- Multilanguage dictionary ---
LANGS = {
    'en': {
        'app_title': '😊 PhoBERT Sentiment Analysis',
        'single_tab': '📝 Single Text Prediction',
        'batch_tab': '📊 Batch Prediction',
        'sidebar_model_info': 'Model Information',
        'sidebar_model': '**Model**: PhoBERT Fine-tuned for Sentiment Analysis',
        'sidebar_labels': '**Labels**: pos (positive), neu (neutral), neg (negative)',
        'sidebar_language': '**Language**: Vietnamese',
        'sidebar_sample_data': 'Sample Data',
        'sidebar_download_sample': '📥 Download Sample CSV',
        'single_section': 'Single Text Prediction',
        'single_input_label': 'Enter Vietnamese text for sentiment analysis:',
        'single_input_placeholder': 'Enter Vietnamese text for sentiment analysis...',
        'single_button': '🔍 Analyze Sentiment',
        'single_pred_label': 'Predicted Label',
        'single_confidence': 'Confidence',
        'single_pos': '😊 Positive',
        'single_neg': '😞 Negative',
        'single_neu': '😐 Neutral',
        'single_original': 'Original Text:',
        'single_warning': 'Please enter some text to analyze.',
        'batch_section': 'Batch Prediction from File',
        'batch_upload_label': 'Upload a CSV file (first column should contain text):',
        'batch_upload_help': 'The first column will be used as input text, regardless of column name.',
        'batch_empty_error': '❌ The uploaded file is empty.',
        'batch_success_load': '✅ Successfully loaded {n} texts from the file.',
        'batch_preview': '📋 Preview of uploaded data',
        'batch_button': '🚀 Run Batch Prediction',
        'batch_processing': 'Processing {n} texts...',
        'batch_success': '✅ Completed analysis of {n} texts!',
        'batch_results': 'Results',
        'batch_total_samples': 'Total Samples',
        'batch_avg_word': 'Avg Word Count',
        'batch_avg_char': 'Avg Char Count',
        'batch_label_dist': 'Label Distribution',
        'batch_download_results': '📥 Download Results as CSV',
        'batch_error': '❌ Error processing file: {err}',
        'batch_file_info': 'Please ensure the file is a valid CSV with text data in the first column.',
        'model_loading': 'Loading PhoBERT model...',
        'model_load_error': '❌ Model could not be loaded. Please check if the model folder exists.',
        'model_loaded': '✅ Model loaded successfully!',
        'theme_label': 'Theme',
        'theme_light': 'Light',
        'theme_dark': 'Dark',
        'lang_label': 'Language',
        'lang_en': 'English',
        'lang_vi': 'Vietnamese',
    },
    'vi': {
        'app_title': '😊 Phân tích cảm xúc PhoBERT',
        'single_tab': '📝 Dự đoán cảm xúc từng câu',
        'batch_tab': '📊 Dự đoán cảm xúc theo tệp',
        'sidebar_model_info': 'Thông tin mô hình',
        'sidebar_model': '**Mô hình**: PhoBERT tinh chỉnh cho phân tích cảm xúc',
        'sidebar_labels': '**Nhãn**: pos (tích cực), neu (trung tính), neg (tiêu cực)',
        'sidebar_language': '**Ngôn ngữ**: Tiếng Việt',
        'sidebar_sample_data': 'Dữ liệu mẫu',
        'sidebar_download_sample': '📥 Tải file CSV mẫu',
        'single_section': 'Dự đoán cảm xúc từng câu',
        'single_input_label': 'Nhập văn bản tiếng Việt để phân tích cảm xúc:',
        'single_input_placeholder': 'Nhập văn bản tiếng Việt để phân tích cảm xúc...',
        'single_button': '🔍 Phân tích cảm xúc',
        'single_pred_label': 'Nhãn dự đoán',
        'single_confidence': 'Độ tin cậy',
        'single_pos': '😊 Tích cực',
        'single_neg': '😞 Tiêu cực',
        'single_neu': '😐 Trung tính',
        'single_original': 'Văn bản gốc:',
        'single_warning': 'Vui lòng nhập văn bản để phân tích.',
        'batch_section': 'Dự đoán cảm xúc theo tệp',
        'batch_upload_label': 'Tải lên file CSV (cột đầu tiên chứa văn bản):',
        'batch_upload_help': 'Chỉ sử dụng cột đầu tiên làm văn bản đầu vào, không cần tên cột.',
        'batch_empty_error': '❌ File tải lên bị rỗng.',
        'batch_success_load': '✅ Đã tải {n} văn bản từ file.',
        'batch_preview': '📋 Xem trước dữ liệu tải lên',
        'batch_button': '🚀 Phân tích cảm xúc hàng loạt',
        'batch_processing': 'Đang xử lý {n} văn bản...',
        'batch_success': '✅ Đã phân tích xong {n} văn bản!',
        'batch_results': 'Kết quả',
        'batch_total_samples': 'Tổng số mẫu',
        'batch_avg_word': 'Số từ TB',
        'batch_avg_char': 'Số ký tự TB',
        'batch_label_dist': 'Phân bố nhãn',
        'batch_download_results': '📥 Tải kết quả CSV',
        'batch_error': '❌ Lỗi xử lý file: {err}',
        'batch_file_info': 'Vui lòng đảm bảo file là CSV hợp lệ và cột đầu tiên chứa văn bản.',
        'model_loading': 'Đang tải mô hình PhoBERT...',
        'model_load_error': '❌ Không thể tải mô hình. Vui lòng kiểm tra thư mục mô hình.',
        'model_loaded': '✅ Đã tải mô hình thành công!',
        'theme_label': 'Giao diện',
        'theme_light': 'Sáng',
        'theme_dark': 'Tối',
        'lang_label': 'Ngôn ngữ',
        'lang_en': 'Tiếng Anh',
        'lang_vi': 'Tiếng Việt',
    }
}

# --- Theme CSS ---
LIGHT_CSS = """
<style>
body, .stApp { background-color: #fff !important; color: #222 !important; }
</style>
"""
DARK_CSS = """
<style>
body, .stApp { background-color: #18191A !important; color: #f1f1f1 !important; }
</style>
"""

# --- Session state for language and theme ---
def get_lang():
    return st.session_state.get('lang', 'en')
def get_theme():
    return st.session_state.get('theme', 'light')

def set_lang(lang):
    st.session_state['lang'] = lang
def set_theme(theme):
    st.session_state['theme'] = theme

# --- UI helpers ---
def _(key, **kwargs):
    lang = get_lang()
    text = LANGS[lang][key]
    if kwargs:
        return text.format(**kwargs)
    return text

# Page configuration
st.set_page_config(
    page_title="PhoBERT Sentiment Analysis",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def create_sample_csv():
    """Create a sample CSV file for download"""
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

def get_download_link(df, filename, text):
    """Generate a download link for a dataframe"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def calculate_statistics(df):
    """Calculate basic statistics for the results"""
    stats = {}
    
    # Label distribution
    label_counts = df['predicted_label'].value_counts()
    stats['label_counts'] = label_counts.to_dict()
    
    # Average sentence length (by word count)
    df['word_count'] = df['text'].str.split().str.len()
    stats['avg_word_count'] = df['word_count'].mean()
    
    # Average character count
    df['char_count'] = df['text'].str.len()
    stats['avg_char_count'] = df['char_count'].mean()
    
    return stats

# --- Main app ---
def main():
    # Sidebar: Language and Theme
    with st.sidebar:
        st.markdown(f"### {_('lang_label')}")
        lang_options = [
            ('en', LANGS['en']['lang_en']),
            ('vi', LANGS['vi']['lang_vi'])
        ]
        lang_labels = [label for code, label in lang_options]
        lang_codes = [code for code, label in lang_options]
        current_lang = get_lang()
        lang_index = lang_codes.index(current_lang) if current_lang in lang_codes else 0
        selected_lang_label = st.selectbox(
            label="",
            options=lang_labels,
            index=lang_index,
            key='lang_selectbox',
        )
        selected_lang_code = lang_codes[lang_labels.index(selected_lang_label)]
        set_lang(selected_lang_code)

        st.markdown(f"### {_('theme_label')}")
        theme_options = [
            ('light', _('theme_light')),
            ('dark', _('theme_dark'))
        ]
        theme_labels = [label for code, label in theme_options]
        theme_codes = [code for code, label in theme_options]
        current_theme = get_theme()
        theme_index = theme_codes.index(current_theme) if current_theme in theme_codes else 0
        selected_theme_label = st.selectbox(
            label="",
            options=theme_labels,
            index=theme_index,
            key='theme_selectbox',
        )
        selected_theme_code = theme_codes[theme_labels.index(selected_theme_label)]
        set_theme(selected_theme_code)

    # Apply theme CSS
    if get_theme() == 'dark':
        st.markdown(DARK_CSS, unsafe_allow_html=True)
    else:
        st.markdown(LIGHT_CSS, unsafe_allow_html=True)

    # Header
    app_title = _('app_title')
    st.markdown(f'<h1 class="main-header">{app_title}</h1>', unsafe_allow_html=True)
    
    # Load model
    with st.spinner(_("model_loading")):
        tokenizer, model = load_model()
    
    if tokenizer is None or model is None:
        st.error(_("model_load_error"))
        return
    
    st.success(_("model_loaded"))
    
    # Sidebar for model info
    with st.sidebar:
        sidebar_model_info = _("sidebar_model_info")
        sidebar_model = _( "sidebar_model")
        sidebar_labels = _( "sidebar_labels")
        sidebar_language = _( "sidebar_language")
        sidebar_sample_data = _( "sidebar_sample_data")
        sidebar_download_sample = _( "sidebar_download_sample")
        st.markdown(f"### {sidebar_model_info}")
        st.info(sidebar_model)
        st.info(sidebar_labels)
        st.info(sidebar_language)
        st.markdown(f"### {sidebar_sample_data}")
        sample_csv = create_sample_csv()
        st.download_button(
            label=sidebar_download_sample,
            data=sample_csv,
            file_name="sample_sentiment_data.csv",
            mime="text/csv"
        )
    
    # Main content
    single_tab = _("single_tab")
    batch_tab = _( "batch_tab")
    tab1, tab2 = st.tabs([single_tab, batch_tab])
    
    with tab1:
        single_section = _( "single_section")
        st.markdown(f'<h2 class="section-header">{single_section}</h2>', unsafe_allow_html=True)
        
        # Text input
        single_input_label = _( "single_input_label")
        single_input_placeholder = _( "single_input_placeholder")
        text_input = st.text_area(
            single_input_label,
            height=150,
            placeholder=single_input_placeholder
        )
        
        single_button = _( "single_button")
        if 'single_predicting' not in st.session_state:
            st.session_state['single_predicting'] = False
        single_disabled = st.session_state['single_predicting']
        if st.button(single_button, type="primary", disabled=single_disabled):
            if text_input.strip():
                st.session_state['single_predicting'] = True
                with st.spinner(_("model_loading")):
                    predicted_label, confidence = predict_sentiment(text_input, tokenizer, model)
                st.session_state['single_predicting'] = False
                col1, col2, col3 = st.columns(3)
                single_pred_label = _( "single_pred_label")
                single_confidence = _( "single_confidence")
                with col1:
                    st.metric(single_pred_label, predicted_label.upper())
                with col2:
                    st.metric(single_confidence, f"{confidence:.2%}")
                with col3:
                    single_pos = _( "single_pos")
                    single_neg = _( "single_neg")
                    single_neu = _( "single_neu")
                    if predicted_label == "pos":
                        st.markdown(f'<div class="success-box">{single_pos}</div>', unsafe_allow_html=True)
                    elif predicted_label == "neg":
                        st.markdown(f'<div class="error-box">{single_neg}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="metric-card">{single_neu}</div>', unsafe_allow_html=True)
                single_original = _( "single_original")
                st.markdown(f"**{single_original}**")
                st.write(text_input)
            else:
                single_warning = _( "single_warning")
                st.warning(single_warning)
    
    with tab2:
        batch_section = _( "batch_section")
        st.markdown(f'<h2 class="section-header">{batch_section}</h2>', unsafe_allow_html=True)
        batch_upload_label = _( "batch_upload_label")
        batch_upload_help = _( "batch_upload_help")
        uploaded_file = st.file_uploader(
            batch_upload_label,
            type=['csv'],
            help=batch_upload_help
        )
        if 'batch_predicting' not in st.session_state:
            st.session_state['batch_predicting'] = False
        batch_disabled = st.session_state['batch_predicting']
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file)
                if df.empty:
                    batch_empty_error = _( "batch_empty_error")
                    st.error(batch_empty_error)
                    return
                text_column = df.iloc[:, 0]
                texts = text_column.astype(str).tolist()
                batch_success_load = _( "batch_success_load", n=len(texts))
                st.success(batch_success_load)
                batch_preview = _( "batch_preview")
                with st.expander(batch_preview):
                    st.dataframe(df.head(10))
                batch_button = _( "batch_button")
                if st.button(batch_button, type="primary", disabled=batch_disabled):
                    st.session_state['batch_predicting'] = True
                    batch_processing = _( "batch_processing", n=len(texts))
                    with st.spinner(batch_processing):
                        results = predict_batch(texts, tokenizer, model, batch_size=32)
                    st.session_state['batch_predicting'] = False
                    results_df = pd.DataFrame({
                        'text': texts,
                        'predicted_label': [result[0] for result in results],
                        'confidence': [result[1] for result in results]
                    })
                    batch_success = _( "batch_success", n=len(results_df))
                    st.success(batch_success)
                    batch_results = _( "batch_results")
                    st.markdown(f"### {batch_results}")
                    st.dataframe(results_df, use_container_width=True)
                    stats = calculate_statistics(results_df)
                    col1, col2, col3 = st.columns(3)
                    batch_total_samples = _( "batch_total_samples")
                    batch_avg_word = _( "batch_avg_word")
                    batch_avg_char = _( "batch_avg_char")
                    with col1:
                        st.metric(batch_total_samples, len(results_df))
                    with col2:
                        st.metric(batch_avg_word, f"{stats['avg_word_count']:.1f}")
                    with col3:
                        st.metric(batch_avg_char, f"{stats['avg_char_count']:.1f}")
                    batch_label_dist = _( "batch_label_dist")
                    st.markdown(f"### {batch_label_dist}")
                    label_counts = pd.Series(stats['label_counts'])
                    fig_pie = px.pie(
                        values=label_counts.values,
                        names=label_counts.index,
                        title=batch_label_dist,
                        color_discrete_map={
                            'pos': '#28a745',
                            'neu': '#ffc107', 
                            'neg': '#dc3545'
                        }
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                    fig_bar = px.bar(
                        x=label_counts.index,
                        y=label_counts.values,
                        title=batch_label_dist,
                        color=label_counts.index,
                        color_discrete_map={
                            'pos': '#28a745',
                            'neu': '#ffc107',
                            'neg': '#dc3545'
                        }
                    )
                    fig_bar.update_layout(xaxis_title="Sentiment", yaxis_title="Count")
                    st.plotly_chart(fig_bar, use_container_width=True)
                    batch_download_results = _( "batch_download_results")
                    st.markdown(f"### {batch_download_results}")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label=batch_download_results,
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                batch_error = _( "batch_error", err=str(e))
                batch_file_info = _( "batch_file_info")
                st.error(batch_error)
                st.info(batch_file_info)

if __name__ == "__main__":
    main() 