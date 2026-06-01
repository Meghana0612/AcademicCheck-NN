"""backend/src/preprocessing.py"""

import re
import spacy

try:
    _nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
except:
    print("Using fallback SpaCy model")
    _nlp = spacy.blank("en")

_EXTRA_SPACES = re.compile(r"\s+")
_NON_TEXT = re.compile(r"[^a-zA-Z0-9\s\.]")


def preprocess_text(text, mode="tfidf"):
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = _NON_TEXT.sub(" ", text)
    text = _EXTRA_SPACES.sub(" ", text).strip()

    if mode == "embed":
        return text  # minimal cleaning for SBERT

    doc = _nlp(text)
    return " ".join(token.lemma_ for token in doc if token.lemma_.strip())


