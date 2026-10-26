import os
import pandas as pd
from sklearn.model_selection import train_test_split
from preprocessing.preprocessing import preprocess_dataframe

# 1. Load raw dataset
raw_csv_path = r"C:\Users\Monika\OneDrive\Desktop\Fake News Detection\data\raw\news.csv"
df = pd.read_csv(raw_csv_path, engine='python', on_bad_lines='skip')

# 2. Drop unused index column (e.g. Unnamed: 0 or empty leading commas)
unnamed_cols = [col for col in df.columns if 'Unnamed' in str(col) or str(col).strip() == '']
if unnamed_cols:
    df = df.drop(columns=unnamed_cols)

# 3. Map 'text' column to 'body' BEFORE preprocessing
if 'text' in df.columns and 'body' not in df.columns:
    df = df.rename(columns={'text': 'body'})

# Ensure missing title/body columns exist as strings
if 'title' not in df.columns:
    df['title'] = ''
if 'body' not in df.columns:
    df['body'] = ''

# 4. Map numeric labels (0 -> REAL, 1 -> FAKE) so preprocessing can validate them
label_map = {0: 'REAL', 1: 'FAKE', '0': 'REAL', '1': 'FAKE'}
if 'label' in df.columns:
    df['label'] = df['label'].map(label_map).fillna(df['label'])

# 5. Clean and preprocess
df = preprocess_dataframe(df)

# 6. Split and save processed CSV files
output_dir = r"C:\Users\Monika\OneDrive\Desktop\Fake News Detection\data\processed"
os.makedirs(output_dir, exist_ok=True)

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, stratify=df['label'])
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42, stratify=temp_df['label'])

train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
val_df.to_csv(os.path.join(output_dir, "validation.csv"), index=False)
test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

# Save ALL articles (REAL and FAKE) for the retrieval corpus; the label is shown next to each retrieved row
retrieval_corpus = df.copy()
retrieval_corpus.to_csv(os.path.join(output_dir, "retrieval_corpus.csv"), index=False)

print("Dataset successfully processed and split into data/processed/")