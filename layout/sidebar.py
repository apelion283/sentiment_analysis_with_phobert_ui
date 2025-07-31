import streamlit as st

def render_sidebar(lang_dict, create_sample_csv, inject_custom_css):
    # Language select
    st.markdown(f"### {lang_dict._('lang_label')}")
    lang_options = [
        ('en', lang_dict.LANGS['en']['lang_en']),
        ('vi', lang_dict.LANGS['vi']['lang_vi'])
    ]
    lang_labels = [label for code, label in lang_options]
    lang_codes = [code for code, label in lang_options]
    current_lang = lang_dict.get_lang()
    lang_index = lang_codes.index(current_lang) if current_lang in lang_codes else 0
    selected_lang_label = st.selectbox(
        label="",
        options=lang_labels,
        index=lang_index,
        key='lang_selectbox',
    )
    selected_lang_code = lang_codes[lang_labels.index(selected_lang_label)]
    lang_dict.set_lang(selected_lang_code)

    # Theme select
    st.markdown(f"### {lang_dict._('theme_label')}")
    theme_options = [
        ('light', lang_dict._('theme_light')),
        ('dark', lang_dict._('theme_dark'))
    ]
    theme_labels = [label for code, label in theme_options]
    theme_codes = [code for code, label in theme_options]
    current_theme = lang_dict.get_theme()
    theme_index = theme_codes.index(current_theme) if current_theme in theme_codes else 0
    selected_theme_label = st.selectbox(
        label="",
        options=theme_labels,
        index=theme_index,
        key='theme_selectbox',
    )
    selected_theme_code = theme_codes[theme_labels.index(selected_theme_label)]
    lang_dict.set_theme(selected_theme_code)

    # Inject custom CSS
    inject_custom_css(lang_dict.get_theme())

    # Model info and sample download
    st.markdown(f"### {lang_dict._('sidebar_model_info')}")
    st.info(lang_dict._('sidebar_model'))
    st.info(lang_dict._('sidebar_labels'))
    st.info(lang_dict._('sidebar_language'))
    st.markdown(f"### {lang_dict._('sidebar_sample_data')}")
    sample_csv = create_sample_csv()
    st.download_button(
        label=lang_dict._('sidebar_download_sample'),
        data=sample_csv,
        file_name="sample_sentiment_data.csv",
        mime="text/csv"
    ) 