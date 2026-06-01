import numpy as np

FEATURE_WEIGHTS = {
    "tfidf_similarity": 0.18,
    "embedding_similarity": 0.30,
    "length_similarity": 0.07,
    "jaccard_similarity": 0.12,
    "char_ngram_similarity": 0.10,
    "token_sort_similarity": 0.08,
    "synonym_jaccard_similarity": 0.07,
    "edit_distance_similarity": 0.04,
    "lcs_similarity": 0.04,
}

def _centered_contribution(score_0_100: float, weight: float) -> float:
    return round(weight * ((score_0_100 - 50.0) / 50.0), 4)

def build_explanations(features: dict):
    shap_values = {}
    lime_weights = []

    for key, weight in FEATURE_WEIGHTS.items():
        val = float(features.get(key, 0.0))
        contrib = _centered_contribution(val, weight)
        shap_values[key] = contrib
        lime_weights.append((key, contrib))

    lime_weights.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "shap_values": {"values": shap_values},
        "lime_weights": lime_weights[:6],
    }

