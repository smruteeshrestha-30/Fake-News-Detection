import os
import re
import joblib
import requests
import urllib.parse
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
from preprocessing.preprocessing import clean_text
from retrieval.bm25_retrieval import BM25Retriever, STOP_WORDS

LIVE_TAG = "Live Web Match"
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}


# --------------------------------------------------------------------------- #
# Live web search
# --------------------------------------------------------------------------- #
def extract_search_keywords(text: str, max_words: int = 8) -> str:
    """Pick the first few distinct, meaningful words to form a clean search query."""
    words = re.findall(r"\b[a-zA-Z]{4,}\b", text or "")
    seen, keywords = set(), []
    for w in words:
        lw = w.lower()
        if lw in STOP_WORDS or lw in seen:
            continue
        seen.add(lw)
        keywords.append(w)
        if len(keywords) >= max_words:
            break
    return " ".join(keywords)


def _google_news_rss(query: str, top_k: int) -> list:
    rss_url = ("https://news.google.com/rss/search?q="
               f"{urllib.parse.quote(query)}&hl=en-US&gl=US&ceid=US:en")
    response = requests.get(rss_url, headers=HEADERS, timeout=8)
    if response.status_code != 200:
        return []

    root = ET.fromstring(response.content)
    results = []
    for item in root.findall('.//item')[:top_k]:
        title_el, link_el = item.find('title'), item.find('link')
        date_el, source_el = item.find('pubDate'), item.find('source')
        title = (title_el.text or 'Untitled') if title_el is not None else 'Untitled'
        source = (source_el.text or '') if source_el is not None else ''
        # Google appends " - Publisher" to titles; strip it when we know the publisher
        if source and title.endswith(f" - {source}"):
            title = title[: -len(f" - {source}")]
        results.append({
            'article_id': 'web',
            'title': title,
            'source': source or 'Live Web Search',
            'date': ((date_el.text or 'N/A')[:16]) if date_el is not None else 'N/A',
            'url': (link_el.text or '#') if link_el is not None else '#',
            'similarity_score': LIVE_TAG,
            'match_type': 'live',
            'same_article': False,
        })
    return results


def fetch_live_web_news(query: str, top_k: int = 5, exclude_title: str = "") -> list:
    """
    Search live news coverage on the web (Google News RSS).
    Returns [] when nothing is found. Tries a long query first, then a shorter one.
    """
    exclude = re.sub(r'\W+', ' ', exclude_title or '').lower().strip()
    for n_words in (8, 5, 3):
        clean_query = extract_search_keywords(query, max_words=n_words)
        if not clean_query:
            return []
        try:
            results = _google_news_rss(clean_query, top_k + 2)
        except Exception as e:
            print(f"[Web Retrieval Error]: {e}")
            return []
        # Drop the article the user submitted, if it shows up in the results
        results = [r for r in results
                   if not exclude or re.sub(r'\W+', ' ', r['title']).lower().strip() != exclude]
        if results:
            return results[:top_k]
    return []


# --------------------------------------------------------------------------- #
# Analyzer
# --------------------------------------------------------------------------- #
def _slug_to_text(url: str) -> str:
    """Turn the last meaningful part of a URL path into words, e.g. /2024/05/mayor-signs-bill -> 'mayor signs bill'."""
    path = urllib.parse.urlparse(url).path
    parts = [p for p in path.split('/') if p]
    for part in reversed(parts):
        part = re.sub(r'\.\w{2,5}$', '', part)
        words = re.findall(r'[A-Za-z]{3,}', part)
        if len(words) >= 3:
            return " ".join(words)
    return ""


class NewsAnalyzer:
    def __init__(self, model_dir=None, corpus_path=None):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        if model_dir is None:
            model_dir = os.path.join(base_dir, "models")
        if corpus_path is None:
            corpus_path = os.path.join(base_dir, "data", "processed", "retrieval_corpus.csv")

        model_path = os.path.join(model_dir, "fake_news_model.joblib")
        vec_path = os.path.join(model_dir, "tfidf_vectorizer.joblib")

        if os.path.exists(model_path) and os.path.exists(vec_path):
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vec_path)
        else:
            self.model = None
            self.vectorizer = None

        if os.path.exists(corpus_path):
            self.retriever = BM25Retriever(corpus_path=corpus_path)
        else:
            self.retriever = None

    def extract_from_url(self, url: str) -> dict:
        """Read a news URL. Returns title/body (body is '' if the page could not be read)."""
        domain = urllib.parse.urlparse(url).netloc or url
        try:
            from newspaper import Article, Config
            cfg = Config()
            cfg.browser_user_agent = HEADERS['User-Agent']
            cfg.request_timeout = 8
            article = Article(url, config=cfg)
            article.download()
            article.parse()
            if article.text and len(article.text.split()) >= 30:
                return {
                    'title': clean_text(article.title),
                    'body': clean_text(article.text),
                    'source': article.source_url or domain,
                    'date': str(article.publish_date) if article.publish_date else 'N/A',
                    'url': url
                }
        except Exception:
            pass

        try:
            resp = requests.get(url, headers=HEADERS, timeout=8)
            soup = BeautifulSoup(resp.text, 'html.parser')
            h1 = soup.find('h1')
            title = h1.get_text() if h1 else (soup.title.get_text() if soup.title else '')
            body = " ".join(p.get_text() for p in soup.find_all('p'))
            return {
                'title': clean_text(title),
                'body': clean_text(body),
                'source': domain,
                'date': 'N/A',
                'url': url
            }
        except Exception:
            return {'title': '', 'body': '', 'source': domain, 'date': 'N/A', 'url': url}

    def _classify(self, full_text: str) -> dict:
        if self.model and self.vectorizer and full_text.strip():
            X_feats = self.vectorizer.transform([full_text])
            probabilities = self.model.predict_proba(X_feats)[0]
            pred_class = self.model.predict(X_feats)[0]
            return {
                'prediction': "LIKELY REAL" if pred_class == 1 else "LIKELY FAKE",
                'prediction_class': 'real' if pred_class == 1 else 'fake',
                'confidence': f"{round(float(probabilities[pred_class]) * 100, 2)}%",
                'real_probability': f"{round(float(probabilities[1]) * 100, 2)}%",
                'fake_probability': f"{round(float(probabilities[0]) * 100, 2)}%",
            }
        label = "CLASSIFIER UNAVAILABLE" if not self.model else "NOT CLASSIFIED (no readable text)"
        return {'prediction': label, 'prediction_class': 'neutral',
                'confidence': "N/A", 'real_probability': "N/A", 'fake_probability': "N/A"}

    def analyze(self, text_input: str = None, url_input: str = None) -> dict:
        metadata = {'title': 'User Submitted Text', 'source': 'Direct Input', 'date': 'N/A', 'url': '#'}
        from_url = bool(url_input and url_input.strip())
        live_query = ""

        if from_url:
            extracted = self.extract_from_url(url_input.strip())
            metadata = {**extracted, 'title': extracted['title'] or 'Could not read article title'}
            full_text = f"{extracted['title']} {extracted['body']}".strip()
            # Query for the live search: headline if we have it, else words from the URL slug
            live_query = extracted['title'] or _slug_to_text(extracted['url'])
            if not full_text:
                full_text = _slug_to_text(extracted['url'])
            if not full_text.strip():
                return {'error': 'Could not read the article at that URL. Try pasting the text instead.'}
            body_readable = bool(extracted['body'].strip())
        else:
            full_text = clean_text(text_input)
            if not full_text.strip():
                return {'error': 'No valid text could be processed.'}
            live_query = full_text[:300]
            body_readable = True

        # Classification (skipped when the URL page gave us no article text)
        classification = self._classify(full_text if body_readable else "")

        # ---- Step 1: is this article from the local corpus? --------------------------
        evidence, evidence_source = [], 'none'
        decision = ""
        if self.retriever:
            hit = self.retriever.find_in_corpus(full_text, top_k=5)
            if hit['in_corpus']:
                evidence, evidence_source = hit['results'], 'corpus'
                decision = ("This article was found in your local corpus (news.csv). "
                            "Showing the most similar corpus articles.")
                same = next((r for r in hit['results'] if r.get('same_article')), None)
                if same and same.get('label') in ('REAL', 'FAKE'):
                    decision += f" The matching article is labelled {same['label']} in the dataset."

        # ---- Step 2: not in corpus -> live web coverage ------------------------------
        if evidence_source == 'none':
            print("[Analyzer]: Local search returned zero matches. Triggering Live Web Search...")
            evidence = fetch_live_web_news(live_query, top_k=5, exclude_title=metadata.get('title', ''))
            if evidence:
                evidence_source = 'live'
                decision = ("Local search returned zero matches, so live news coverage "
                            "was pulled from the web.")
            else:
                decision = "Not found in the local corpus and no live web coverage was found."
                print("[Analyzer]: Nothing found locally or on the web.")

        return {
            'article_info': metadata,
            'input_type': 'url' if from_url else 'text',
            **classification,
            'evidence': evidence,
            'evidence_source': evidence_source,
            'decision': decision,
            'limitation_warning': (
                "This is an automated prediction, not a definitive fact-check. "
                "Retrieved articles are supporting evidence and should be independently verified."
            )
        }
