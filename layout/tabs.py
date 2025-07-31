import streamlit as st
import pandas as pd
import plotly.express as px

def render_tabs(lang_dict, predict_sentiment, predict_batch, calculate_statistics):
    single_tab = lang_dict._("single_tab")
    batch_tab = lang_dict._("batch_tab")
    tab1, tab2 = st.tabs([single_tab, batch_tab])

    with tab1:
        single_section = lang_dict._("single_section")
        st.markdown(f'<h2 class="section-header">{single_section}</h2>', unsafe_allow_html=True)
        single_input_label = lang_dict._("single_input_label")
        single_input_placeholder = lang_dict._("single_input_placeholder")
        text_input = st.text_area(
            single_input_label,
            height=150,
            placeholder=single_input_placeholder
        )
        single_button = lang_dict._("single_button")
        if 'single_predicting' not in st.session_state:
            st.session_state['single_predicting'] = False
        single_disabled = st.session_state['single_predicting']
        if st.button(single_button, type="primary", disabled=single_disabled):
            if text_input.strip():
                st.session_state['single_predicting'] = True
                with st.spinner(lang_dict._("model_loading")):
                    predicted_label, confidence = predict_sentiment(text_input)
                st.session_state['single_predicting'] = False
                col1, col2, col3 = st.columns(3)
                single_pred_label = lang_dict._("single_pred_label")
                single_confidence = lang_dict._("single_confidence")
                with col1:
                    st.metric(single_pred_label, predicted_label.upper())
                with col2:
                    st.metric(single_confidence, f"{confidence:.2%}")
                with col3:
                    single_pos = lang_dict._("single_pos")
                    single_neg = lang_dict._("single_neg")
                    single_neu = lang_dict._("single_neu")
                    if predicted_label == "pos":
                        st.markdown(f'<div class="success-box">{single_pos}</div>', unsafe_allow_html=True)
                    elif predicted_label == "neg":
                        st.markdown(f'<div class="error-box">{single_neg}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="metric-card">{single_neu}</div>', unsafe_allow_html=True)
                single_original = lang_dict._("single_original")
                st.markdown(f"**{single_original}**")
                st.write(text_input)
            else:
                single_warning = lang_dict._("single_warning")
                st.warning(single_warning)

    with tab2:
        batch_section = lang_dict._("batch_section")
        st.markdown(f'<h2 class="section-header">{batch_section}</h2>', unsafe_allow_html=True)
        batch_upload_label = lang_dict._("batch_upload_label")
        batch_upload_help = lang_dict._("batch_upload_help")
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
                    batch_empty_error = lang_dict._("batch_empty_error")
                    st.error(batch_empty_error)
                    return
                text_column = df.iloc[:, 0]
                texts = text_column.astype(str).tolist()
                batch_success_load = lang_dict._("batch_success_load", n=len(texts))
                st.success(batch_success_load)
                batch_preview = lang_dict._("batch_preview")
                with st.expander(batch_preview):
                    st.dataframe(df.head(10))
                batch_button = lang_dict._("batch_button")
                if st.button(batch_button, type="primary", disabled=batch_disabled):
                    st.session_state['batch_predicting'] = True
                    batch_processing = lang_dict._("batch_processing", n=len(texts))
                    with st.spinner(batch_processing):
                        results = predict_batch(texts)
                    st.session_state['batch_predicting'] = False
                    results_df = pd.DataFrame({
                        'text': texts,
                        'predicted_label': [result[0] for result in results],
                        'confidence': [result[1] for result in results]
                    })
                    batch_success = lang_dict._("batch_success", n=len(results_df))
                    st.success(batch_success)
                    batch_results = lang_dict._("batch_results")
                    st.markdown(f"### {batch_results}")
                    st.dataframe(results_df, use_container_width=True)
                    stats = calculate_statistics(results_df)
                    col1, col2, col3 = st.columns(3)
                    batch_total_samples = lang_dict._("batch_total_samples")
                    batch_avg_word = lang_dict._("batch_avg_word")
                    batch_avg_char = lang_dict._("batch_avg_char")
                    with col1:
                        st.metric(batch_total_samples, len(results_df))
                    with col2:
                        st.metric(batch_avg_word, f"{stats['avg_word_count']:.1f}")
                    with col3:
                        st.metric(batch_avg_char, f"{stats['avg_char_count']:.1f}")
                    batch_label_dist = lang_dict._("batch_label_dist")
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
                    batch_download_results = lang_dict._("batch_download_results")
                    st.markdown(f"### {batch_download_results}")
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label=batch_download_results,
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                batch_error = lang_dict._("batch_error", err=str(e))
                batch_file_info = lang_dict._("batch_file_info")
                st.error(batch_error)
                st.info(batch_file_info) 