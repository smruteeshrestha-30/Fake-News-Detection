import re
import html
import pandas as pd

def clean_text(text: str) -> str:
    """Cleans raw text by removing HTML, normalizing spaces, and unescaping special characters."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = html.unescape(text)
    text = re.sub(r'<[^>]+>', ' ', text)  # Remove HTML tags
    text = re.sub(r'\s+', ' ', text)     # Normalize whitespace
    return text.strip()

def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Filters duplicates, missing values, and prepares title+body text for classification."""
    df = df.copy()
    
    # Ensure label standardization
    df['label'] = df['label'].astype(str).str.upper().str.strip()
    df = df[df['label'].isin(['REAL', 'FAKE'])].copy()
    
    # Fill missing fields
    df['title'] = df['title'].fillna('').apply(clean_text)
    df['body'] = df['body'].fillna('').apply(clean_text)
    
    # Filter short articles
    df['full_text'] = (df['title'] + " " + df['body']).str.strip()
    df = df[df['full_text'].str.len() > 50]
    
    # Remove exact duplicates on content
    df = df.drop_duplicates(subset=['full_text'])
    
    return df