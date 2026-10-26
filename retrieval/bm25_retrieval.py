import re
import math
import numpy as np
import pandas as pd
from collections import Counter
from rank_bm25 import BM25Okapi

STOP_WORDS = set([
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
    'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
    'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
    'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
    'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
    'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
    'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
    'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
    'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now',
    'said', 'says', 'new', 'also', 'one', 'two', 'time', 'year', 'years', 'first', 'like'
])

def tokenize(text: str) -> list:
    """Extract clean lowercased alphabetic tokens, removing stop words."""
    if not isinstance(text, str):
        return []
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    return [w for w in words if w not in STOP_WORDS]


def normalize_words(text: str) -> list:
    """Lowercased alphanumeric words with NO stop-word removal (used for exact-copy checks)."""
    if not isinstance(text, str):
        return []
    return re.findall(r"[a-z0-9]+", text.lower())


def _shingles(words: list, n: int = 5) -> set:
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


class BM25Retriever:
    # A pasted article is treated as "from the corpus" when at least this share of its
    # word 5-grams (or, for short headlines, its words) is found inside one corpus article.
    IN_CORPUS_THRESHOLD = 0.5
    SHORT_THRESHOLD = 0.9      # headlines: exact phrase, or (almost) every word present
    SHORT_QUERY_WORDS = 15
    MAX_QUERY_TOKENS = 80      # BM25 query size cap (keeps scoring fast)
    CANDIDATES = 10            # BM25 candidates that get verified

    def __init__(self, corpus_path="data/processed/retrieval_corpus.csv"):
        # Only load the columns we need (the processed file also stores a duplicate full_text column)
        wanted = ['id', 'article_id', 'title', 'body', 'text', 'label', 'url', 'source', 'date']
        header = pd.read_csv(corpus_path, nrows=0).columns
        usecols = [c for c in header if c in wanted]
        self.corpus_df = pd.read_csv(corpus_path, usecols=usecols).fillna("")

        self.title_col = 'title' if 'title' in self.corpus_df.columns else ''
        self.text_col = 'body' if 'body' in self.corpus_df.columns else ('text' if 'text' in self.corpus_df.columns else '')

        combined_text = self._combined_series()
        tokenized_corpus = [tokenize(doc) for doc in combined_text]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _combined_series(self):
        parts = []
        if self.title_col:
            parts.append(self.corpus_df[self.title_col].astype(str))
        if self.text_col:
            parts.append(self.corpus_df[self.text_col].astype(str))
        if not parts:
            return pd.Series([""] * len(self.corpus_df))
        out = parts[0]
        for extra in parts[1:]:
            out = out + " " + extra
        return out

    def _doc_text(self, idx: int) -> str:
        row = self.corpus_df.iloc[idx]
        return f"{row.get(self.title_col, '')} {row.get(self.text_col, '')}"

    def _row_to_result(self, idx: int, similarity: float, same_article: bool) -> dict:
        row = self.corpus_df.iloc[idx]
        article_id = row.get('article_id', row.get('id', idx))
        return {
            'article_id': article_id,
            'title': row.get('title', 'Untitled') or 'Untitled',
            'source': row.get('source', '') or 'Local corpus (news.csv)',
            'date': row.get('date', '') or 'N/A',
            'url': row.get('url', '') or '#',
            'similarity_score': f"{round(similarity * 100, 1)}%",
            'match_type': 'corpus',
            'label': str(row.get('label', '')).upper(),
            'same_article': same_article,
        }

    # ------------------------------------------------------------------ #
    def containment(self, query_words: list, doc_text: str, doc_title: str = "") -> float:
        """How much of the query is literally contained in the document (0..1)."""
        doc_words = normalize_words(doc_text)
        if not query_words or not doc_words:
            return 0.0

        if len(query_words) < self.SHORT_QUERY_WORDS:
            # Headline / short claim: exact phrase in the article, or (almost) the whole title
            if " ".join(query_words) in " ".join(doc_words):
                return 1.0
            title_set = set(normalize_words(doc_title))
            return sum(w in title_set for w in query_words) / len(query_words)

        q_sh = _shingles(query_words[:400])
        if not q_sh:
            return 0.0
        return len(q_sh & _shingles(doc_words)) / len(q_sh)

    def similarity(self, query_tokens: list, doc_text: str, short: bool) -> float:
        """Display similarity: query coverage for short queries, TF cosine for long ones."""
        doc_tokens = tokenize(doc_text)
        if not query_tokens or not doc_tokens:
            return 0.0
        if short:
            doc_set = set(doc_tokens)
            return sum(t in doc_set for t in query_tokens) / len(query_tokens)
        qc, dc = Counter(query_tokens), Counter(doc_tokens)
        dot = sum(qc[t] * dc.get(t, 0) for t in qc)
        norm = math.sqrt(sum(v * v for v in qc.values())) * math.sqrt(sum(v * v for v in dc.values()))
        return dot / norm if norm else 0.0

    def find_in_corpus(self, query: str, top_k: int = 5) -> dict:
        """
        Decide whether `query` (pasted text or extracted article) comes from the corpus.
        Returns {'in_corpus': bool, 'results': [...]}.
        results are the most similar corpus articles, only when in_corpus is True.
        """
        q_tokens_all = tokenize(query)
        if not q_tokens_all:
            return {'in_corpus': False, 'results': []}

        q_words = normalize_words(query)
        short = len(q_words) < self.SHORT_QUERY_WORDS
        q_tokens = q_tokens_all[: self.MAX_QUERY_TOKENS]

        scores = self.bm25.get_scores(q_tokens)
        n = min(self.CANDIDATES, len(scores))
        cand_idx = np.argpartition(scores, -n)[-n:]
        cand_idx = [int(i) for i in cand_idx if scores[i] > 0]
        cand_idx.sort(key=lambda i: scores[i], reverse=True)

        scored = []
        for i in cand_idx:
            text = self._doc_text(i)
            title = str(self.corpus_df.iloc[i].get(self.title_col, ''))
            scored.append((i, self.containment(q_words, text, title), self.similarity(q_tokens_all, text, short)))

        threshold = self.SHORT_THRESHOLD if short else self.IN_CORPUS_THRESHOLD
        if not scored or max(c for _, c, _ in scored) < threshold:
            return {'in_corpus': False, 'results': []}

        # Same article(s) first, then by similarity
        scored.sort(key=lambda t: (t[1] >= threshold, t[2]), reverse=True)
        results = [
            self._row_to_result(i, sim, same_article=(cont >= threshold))
            for i, cont, sim in scored[:top_k]
        ]
        return {'in_corpus': True, 'results': results}
