"""Matching and evaluation. Scores measure text similarity, never hiring suitability."""
from __future__ import annotations
import re
import time
from io import BytesIO
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
MAX_CHARS = 50000


def extract_text(name: str, data: bytes) -> str:
    if len(data) > 5 * 1024 * 1024:
        raise ValueError('File exceeds the 5 MB limit.')
    suffix = Path(name).suffix.lower()
    if suffix == '.txt':
        text = data.decode('utf-8-sig')
    elif suffix == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise ValueError('Use an unencrypted PDF.')
        if len(reader.pages) > 30:
            raise ValueError('Maximum 30 PDF pages per resume.')
        pages = []
        for number, page in enumerate(reader.pages, 1):
            page_text = page.extract_text() or ''
            if not page_text.strip() and len(page.images):
                raise ValueError(f'Page {number} contains an image but no readable text. Scanned resumes are unsupported; upload a text-based PDF, Word or TXT document.')
            pages.append(page_text)
        text = '\n'.join(pages)
    elif suffix == '.docx':
        from docx import Document
        doc = Document(BytesIO(data))
        parts = [p.text for p in doc.paragraphs]
        parts += [' | '.join(c.text for c in r.cells) for t in doc.tables for r in t.rows]
        text = '\n'.join(parts)
    else:
        raise ValueError('Use TXT, DOCX or a text-based PDF.')
    if not text.strip():
        raise ValueError('No readable text found. Scanned resumes and images are unsupported; upload a text-based PDF, Word or TXT document.')
    if len(text) > MAX_CHARS:
        raise ValueError('Extracted text exceeds 50,000 characters; shorten the document.')
    return text


def clean_text(text: str) -> str:
    """Limited redaction, not anonymization. Preview and edit before matching."""
    lines = []
    for line in text.splitlines():
        if re.match(r'^\s*(name|candidate|email|phone|address|date of birth|gender|nationality)\s*:', line, re.I):
            continue
        line = re.sub(r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}', '', line)
        line = re.sub(r'https?://\S+', '', line)
        line = re.sub(r'(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)', '', line)
        if line.strip():
            lines.append(line.strip())
    return '\n'.join(lines)


def chunks(text: str, size: int = 80) -> list[str]:
    """Keep short lines intact; overlap long lines so later pages are represented."""
    out = []
    for paragraph in text.splitlines():
        words = paragraph.strip().split()
        for start in range(0, len(words), size - 15):
            out.append(' '.join(words[start:start + size]))
            if start + size >= len(words):
                break
    return [x for x in out if x]


def match(criteria: list[str], resumes: dict[str, str], method: str, model=None):
    if not criteria or not all(c.strip() for c in criteria):
        raise ValueError('Enter at least one non-empty job criterion.')
    if not resumes or any(not t.strip() for t in resumes.values()):
        raise ValueError('Every candidate needs readable text.')
    if method not in ('keyword', 'semantic'):
        raise ValueError('Unknown method.')
    start = time.perf_counter()
    ids, all_chunks, spans = list(resumes), [], []
    for cid in ids:
        parts = chunks(resumes[cid])
        spans.append((len(all_chunks), len(all_chunks) + len(parts)))
        all_chunks.extend(parts)
    corpus = criteria + all_chunks
    if method == 'keyword':
        # Preserve short technical terms and C++/C#; same evidence units as semantic.
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2),
                                    token_pattern=r'(?u)\b\w[\w+#.-]*', sublinear_tf=True)
        try:
            vectors = vectorizer.fit_transform(corpus)
        except ValueError as exc:
            raise ValueError('No usable words found in the inputs.') from exc
        similarity = (vectors[:len(criteria)] @ vectors[len(criteria):].T).toarray()
    else:
        if model is None:
            raise ValueError('Load the Sentence Transformer model before semantic matching.')
        vectors = model.encode(corpus, normalize_embeddings=True, convert_to_numpy=True,
                               show_progress_bar=False, batch_size=32)
        similarity = vectors[:len(criteria)] @ vectors[len(criteria):].T
    rows, evidence = [], []
    for cid, (lo, hi) in zip(ids, spans):
        local = similarity[:, lo:hi]
        best = local.argmax(axis=1)
        scores = local[np.arange(len(criteria)), best]
        rows.append({'candidate_id': cid, 'score': float(scores.mean())})
        for i, criterion in enumerate(criteria):
            evidence.append({'candidate_id': cid, 'criterion': criterion,
                             'similarity': float(scores[i]),
                             'supporting_text': all_chunks[lo + int(best[i])]})
    rows.sort(key=lambda r: (-r['score'], r['candidate_id']))
    for rank, row in enumerate(rows, 1):
        row['rank'] = rank
    return {'ranking': rows, 'evidence': evidence,
            'seconds': time.perf_counter() - start, 'method': method}


def evaluate(ranking: list[dict], labels: dict[str, int], k: int = 5):
    ids = [r['candidate_id'] for r in ranking]
    if set(ids) != set(labels):
        raise ValueError('Labels must contain exactly one label for every ranked candidate.')
    if any(v not in (0, 1, 2) for v in labels.values()):
        raise ValueError('Relevance labels must be 0, 1 or 2.')
    if not 1 <= k <= len(ids):
        raise ValueError('k must be between 1 and the number of candidates.')
    hits = sum(labels[cid] == 2 for cid in ids[:k])
    relevant = sum(v == 2 for v in labels.values())
    return {'k': k, 'precision': hits / k,
            'recall': hits / relevant if relevant else None,
            'relevant_total': relevant, 'hits': hits}


def safe_csv(frame):
    """Prevent spreadsheet formula interpretation of untrusted document text."""
    frame = frame.copy()
    for col in frame.columns:
        frame[col] = frame[col].map(lambda x: "'" + x if isinstance(x, str)
                                  and x.lstrip().startswith(('=', '+', '-', '@')) else x)
    return frame.to_csv(index=False).encode('utf-8-sig')
