"""
Build the retrieval corpus from ALL articles (REAL and FAKE) in news.csv.

Combines the already-processed train/validation/test splits, so:
  - no need to re-process the raw file,
  - the train/test split (and your trained model) stay exactly as they are.

Run once:  python build_retrieval_corpus.py
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
PROCESSED = os.path.join(BASE, "data", "processed")

parts = [pd.read_csv(os.path.join(PROCESSED, f"{name}.csv"))
         for name in ("train", "validation", "test")]
corpus = pd.concat(parts, ignore_index=True)

# Keep only what retrieval needs (full_text duplicates title+body and just wastes memory)
keep = [c for c in ("id", "title", "body", "label") if c in corpus.columns]
corpus = corpus[keep].drop_duplicates(subset=[c for c in ("title", "body") if c in keep])

out = os.path.join(PROCESSED, "retrieval_corpus.csv")
corpus.to_csv(out, index=False)
print(f"Saved {len(corpus)} articles to {out}")
print(corpus['label'].value_counts().to_string() if 'label' in corpus.columns else "")
