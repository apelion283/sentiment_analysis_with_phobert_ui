import streamlit as st

def render_header(lang_dict):
    app_title = lang_dict._('app_title')
    st.markdown(f'<h1 class="main-header">{app_title}</h1>', unsafe_allow_html=True) 