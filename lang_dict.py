import streamlit as st

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

def get_lang():
    return st.session_state.get('lang', 'en')

def set_lang(lang):
    st.session_state['lang'] = lang

def get_theme():
    return st.session_state.get('theme', 'light')

def set_theme(theme):
    st.session_state['theme'] = theme

def _(key, **kwargs):
    lang = get_lang()
    text = LANGS[lang][key]
    if kwargs:
        return text.format(**kwargs)
    return text 