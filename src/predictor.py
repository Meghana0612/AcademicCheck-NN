import os
import json
import pickle
import numpy as np

from sklearn.preprocessing import normalize
from sentence_transformers import CrossEncoder

from src.preprocessing import preprocess_text
from src.embedding import embed_batch
from src.similarity import (
    build_tfidf_matrix,
    tfidf_pair_similarity,
    length_similarity,
    jaccard_similarity,
    char_ngram_similarity,
    token_sort_similarity,
    synonym_jaccard_similarity,
    edit_distance_similarity,
    lcs_similarity,
    highlight_similar_sentences,
)
from src.explainability import build_explanations


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


class PlagiarismPredictor:
    def __init__(self, model_dir: str = "models"):
        self._ready = False
        self.best_threshold = 0.5
        self.tfidf_vec = None
        self._load(model_dir)

    def _load(self, model_dir: str):
        ce_path = os.path.join(model_dir, "cross_encoder")
        tfidf_path = os.path.join(model_dir, "tfidf_vectorizer.pkl")

        if not os.path.isdir(ce_path):
            print("[predictor] WARNING: cross_encoder not found. Run train.py first.")
            return

        self.model = CrossEncoder(ce_path, num_labels=1)

        config_path = os.path.join(ce_path, "train_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                self.best_threshold = json.load(f).get("best_threshold", 0.5)

        if os.path.exists(tfidf_path):
            with open(tfidf_path, "rb") as f:
                self.tfidf_vec = pickle.load(f)

        self._ready = True
        print(f"[predictor] Loaded CrossEncoder. Threshold: {self.best_threshold}")

    def is_ready(self) -> bool:
        return self._ready

    def _display_features(self, text1: str, text2: str) -> dict:
        p1c = preprocess_text(text1, "tfidf")
        p2c = preprocess_text(text2, "tfidf")
        p1r = preprocess_text(text1, "embed")
        p2r = preprocess_text(text2, "embed")

        tfidf_sim = 0.0
        if self.tfidf_vec is not None:
            _, mat = build_tfidf_matrix([p1c, p2c], self.tfidf_vec)
            tfidf_sim = tfidf_pair_similarity(mat[0], mat[1])

        emb = normalize(embed_batch([p1r, p2r]))
        embed_sim = float(np.dot(emb[0], emb[1]))

        return {
            "tfidf_similarity": round(tfidf_sim * 100, 2),
            "embedding_similarity": round(embed_sim * 100, 2),
            "length_similarity": round(length_similarity(p1c, p2c) * 100, 2),
            "jaccard_similarity": round(jaccard_similarity(p1c, p2c) * 100, 2),
            "char_ngram_similarity": round(char_ngram_similarity(p1c, p2c) * 100, 2),
            "token_sort_similarity": round(token_sort_similarity(p1c, p2c) * 100, 2),
            "synonym_jaccard_similarity": round(synonym_jaccard_similarity(p1c, p2c) * 100, 2),
            "edit_distance_similarity": round(edit_distance_similarity(p1c, p2c) * 100, 2),
            "lcs_similarity": round(lcs_similarity(p1c, p2c) * 100, 2),
        }

    def predict(self, text1: str, text2: str) -> dict:
        if not self._ready:
            raise RuntimeError("Model not loaded. Run train.py first.")

        raw = float(self.model.predict([(text1, text2)])[0])
        prob = float(_sigmoid(raw))
        label = 1 if prob >= self.best_threshold else 0

        display = self._display_features(text1, text2)
        explain = build_explanations(display)
        matches = highlight_similar_sentences(text1, text2)

        return {
            "verdict": "Plagiarized" if label == 1 else "Not Plagiarized",
            "probability": round(prob * 100, 2),
            "threshold": round(self.best_threshold, 2),
            **display,
            **explain,
            "matched_sentences": matches,
        }
    

    