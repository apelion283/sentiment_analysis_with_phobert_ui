import streamlit as st
import lang_dict
from layout.header import render_header
from layout.sidebar import render_sidebar
from layout.tabs import render_tabs
import utils

def main():
    st.set_page_config(
        page_title="PhoBERT Sentiment Analysis",
        page_icon="😊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    # Sidebar
    with st.sidebar:
        render_sidebar(lang_dict, utils.create_sample_csv, utils.inject_custom_css)
    # Header
    render_header(lang_dict)
    # Load model (show spinner, error if not loaded)
    with st.spinner(lang_dict._("model_loading")):
        tokenizer, model = utils.get_model()
    if tokenizer is None or model is None:
        st.error(lang_dict._("model_load_error"))
        return
    st.success(lang_dict._("model_loaded"))
    # Tabs
    render_tabs(
        lang_dict,
        utils.predict_sentiment_wrapper,
        utils.predict_batch_wrapper,
        utils.calculate_statistics
    )

if __name__ == "__main__":
    main() 