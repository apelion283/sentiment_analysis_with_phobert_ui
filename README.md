# PhoBERT Sentiment Analysis Web Application

A Streamlit web application for Vietnamese sentiment analysis using a fine-tuned PhoBERT model. The application can classify text into three sentiment categories: positive (pos), neutral (neu), and negative (neg).

## Features

- **Single Text Prediction**: Analyze sentiment for individual Vietnamese texts
- **Batch Prediction**: Process multiple texts from CSV files
- **Interactive Visualizations**: Pie charts and bar charts showing sentiment distribution
- **Statistics**: Average sentence length and label distribution statistics
- **Download Results**: Export analysis results as CSV files
- **Sample Data**: Download sample CSV file for testing

## Requirements

- Python 3.8+
- PhoBERT fine-tuned model in `phobert_sentiment_model/` folder

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your PhoBERT model**:
   - Place your fine-tuned PhoBERT model files in a folder named `phobert_sentiment_model/`
   - The folder should contain:
     - `config.json`
     - `pytorch_model.bin` (or model files)
     - `tokenizer.json` (or tokenizer files)
     - `vocab.txt`

## Usage

### Running the Application

1. **Start the Streamlit app**:
   ```bash
   streamlit run app.py
   ```

2. **Open your browser** and navigate to the URL shown in the terminal (usually `http://localhost:8501`)

### Using the Application

#### Single Text Prediction
1. Go to the "📝 Single Text Prediction" tab
2. Enter Vietnamese text in the text area
3. Click "🔍 Analyze Sentiment"
4. View the predicted sentiment label and confidence score

#### Batch Prediction
1. Go to the "📊 Batch Prediction" tab
2. Upload a CSV file with text data in the first column
3. Click "🚀 Run Batch Prediction"
4. View results, statistics, and visualizations
5. Download results as CSV

### CSV File Format

For batch prediction, your CSV file should have:
- Text data in the **first column** (column name doesn't matter)
- One text per row
- No header required

Example:
```csv
Tôi rất thích sản phẩm này!
Dịch vụ khách hàng không tốt.
Sản phẩm bình thường.
```

## Model Information

- **Model**: PhoBERT (Vietnamese BERT)
- **Task**: Sentiment Classification
- **Labels**: 
  - `neg` (negative) - Label index 0
  - `neu` (neutral) - Label index 1  
  - `pos` (positive) - Label index 2
- **Language**: Vietnamese

## Features in Detail

### Single Text Analysis
- Real-time sentiment prediction
- Confidence scores
- Color-coded results (green for positive, yellow for neutral, red for negative)

### Batch Analysis
- Efficient batch processing (batch size: 32)
- Progress indicators
- Comprehensive statistics:
  - Total number of samples
  - Average word count
  - Average character count
  - Label distribution

### Visualizations
- **Pie Chart**: Shows percentage distribution of sentiments
- **Bar Chart**: Shows count distribution of sentiments
- Color-coded charts for easy interpretation

### Error Handling
- Invalid file format detection
- Empty file handling
- Model loading error handling
- User-friendly error messages

## File Structure

```
Sentiment_Analysis_UI/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── phobert_sentiment_model/  # Your model folder (not included)
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    └── vocab.txt
```

## Troubleshooting

### Model Loading Issues
- Ensure the `phobert_sentiment_model/` folder exists
- Check that all required model files are present
- Verify the model is compatible with the transformers library version

### File Upload Issues
- Ensure the uploaded file is a valid CSV
- Check that the first column contains text data
- Verify the file is not empty

### Performance Issues
- For large files, the application processes data in batches of 32
- Consider splitting very large files if processing is slow

## Customization

### Changing Batch Size
Modify the `batch_size` parameter in the `predict_batch` function call:
```python
results = predict_batch(texts, tokenizer, model, batch_size=64)
```

### Adding New Visualizations
The application uses Plotly for visualizations. You can add new charts by creating additional Plotly figures and using `st.plotly_chart()`.

### Modifying Label Mapping
If your model uses different label indices, update the `label_mapping` dictionary:
```python
label_mapping = {0: "neg", 1: "neu", 2: "pos"}
```

## Contributing

Feel free to submit issues and enhancement requests!

## License

This project is open source and available under the MIT License. 