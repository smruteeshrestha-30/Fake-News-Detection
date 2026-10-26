# Fake News Detection Using Machine Learning and Information Retrieval

## Overview

Fake News Detection is a machine learning project that analyzes news text and predicts whether it is likely to be **Fake** or **Real**. The project combines text preprocessing, TF-IDF feature extraction, machine learning classification, and BM25-based information retrieval in a web application.

The application allows users to submit news text and view the classification results through a simple web interface.

## Features

- **News Classification:** Predicts whether news text is likely to be fake or real.
- **Text Preprocessing:** Cleans and prepares textual data for analysis.
- **TF-IDF Vectorization:** Converts text into numerical features for machine learning.
- **BM25 Information Retrieval:** Retrieves relevant content from a news corpus.
- **Web Interface:** Enables users to submit text and view analysis results.
- **Modular Design:** Separates classification, preprocessing, retrieval, and application logic.
- **Pretrained Models:** Includes the trained classifier and TF-IDF vectorizer.

## Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Machine Learning | Scikit-learn |
| Text Feature Extraction | TF-IDF |
| Information Retrieval | BM25 |
| Web Framework | Flask |
| Data Processing | Pandas, NumPy |
| Model Serialization | Joblib |
| Frontend | HTML, CSS |
| Version Control | Git, GitHub |

## Project Structure

```text
Fake-News-Detection/
├── app.py
├── prepare_data.py
├── build_retrieval_corpus.py
├── classification/
│   └── train.py
├── data/
│   ├── raw/
│   └── processed/
├── models/
│   ├── fake_news_model.joblib
│   └── tfidf_vectorizer.joblib
├── pipeline/
│   └── analyzer.py
├── preprocessing/
│   └── preprocessing.py
├── retrieval/
│   └── bm25_retrieval.py
├── static/
│   └── style.css
├── templates/
│   ├── index.html
│   └── result.html
├── requirements.txt
├── .gitignore
└── README.md
```

## How It Works

1. **User Input:** The user submits news text through the web interface.
2. **Preprocessing:** The input text is cleaned and prepared for analysis.
3. **Feature Extraction:** The TF-IDF vectorizer transforms the text into numerical features.
4. **Classification:** The trained machine learning model predicts the likely news category.
5. **Information Retrieval:** The BM25 component retrieves relevant content from the available news corpus.
6. **Result Presentation:** The application displays the analysis results through the web interface.

## Installation and Setup

### Prerequisites

- Python
- pip
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/smruteeshrestha-30/Fake-News-Detection.git
cd Fake-News-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```bash
venv\Scripts\activate
```

On macOS or Linux:

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the Datasets

Large CSV datasets are excluded from this repository to keep the repository lightweight and comply with GitHub file-size limits.

Place the required datasets in the appropriate `data/raw/` and `data/processed/` directories before running scripts that require them. Ensure that the dataset format and column names match the requirements of the project scripts.

The trained model and TF-IDF vectorizer are available in the `models/` directory, so retraining may not be necessary if the existing model files are compatible with the application.

### 5. Run the Application

```bash
python app.py
```

Open the local URL displayed in your terminal. For a standard Flask development server, this is commonly:

`http://127.0.0.1:5000/`

## Model and Evaluation

The project uses a trained machine learning classifier with TF-IDF text features. It also incorporates BM25-based information retrieval to identify relevant news content.

The model should be evaluated on held-out test data using metrics such as:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Actual evaluation results should be reported after running the evaluation on the test dataset. Performance may vary depending on the dataset, preprocessing pipeline, and distribution of news content.

## Limitations

- Predictions depend on the quality and representativeness of the training data.
- The model may misclassify genuine or misleading news.
- Retrieved content and textual similarity do not independently establish whether a claim is true.
- The system should be used as an analytical aid rather than a definitive fact-checking authority.

## Future Improvements

- Integrate trusted external fact-checking sources.
- Improve model explainability.
- Expand testing across different news topics and datasets.
- Add detailed evaluation visualizations.
- Deploy the application to a cloud platform.
- Incorporate source credibility and claim verification.

## Responsible Use

This project is intended for educational and research purposes. Its predictions should be independently verified before being used to make consequential decisions about news sources or individuals.

## Author

**Smrutee Shrestha**

MSc Data Science

GitHub: [smruteeshrestha-30](https://github.com/smruteeshrestha-30)

---

If you find this project useful, feel free to explore the code and suggest improvements.
