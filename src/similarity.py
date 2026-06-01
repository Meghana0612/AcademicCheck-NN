"""backend/src/similarity.py"""

from functools import lru_cache

import numpy as np
import pandas as pd
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from sklearn.preprocessing import normalize
from scipy.optimize import linear_sum_assignment

from nltk.tokenize import sent_tokenize
from nltk.corpus import wordnet

from src.embedding import embed_batch
from src.preprocessing import preprocess_text

nltk.download("punkt",      quiet=True)
nltk.download("punkt_tab",  quiet=True)
nltk.download("wordnet",    quiet=True)
nltk.download("omw-1.4",    quiet=True)


# ==============================
# TF-IDF
# ==============================

def build_tfidf_matrix(corpus, vectorizer=None):
    if vectorizer is None:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=20000,
            min_df=2,
            max_df=0.9,
        )
        matrix = vectorizer.fit_transform(corpus)
    else:
        matrix = vectorizer.transform(corpus)
    return vectorizer, matrix


def tfidf_pair_similarity(vec1, vec2):
    return float(sk_cosine(vec1, vec2)[0][0])


# ==============================
# ORIGINAL FEATURES
# ==============================

def length_similarity(t1, t2):
    l1 = max(len(t1.split()), 1)
    l2 = max(len(t2.split()), 1)
    return min(l1, l2) / max(l1, l2)


def jaccard_similarity(t1, t2):
    s1 = set(t1.split())
    s2 = set(t2.split())
    if not s1 or not s2:
        return 0.0
    return len(s1 & s2) / len(s1 | s2)


def char_ngram_similarity(t1, t2, n=3):
    def get_ngrams(text):
        text = text.replace(" ", "")
        return set(text[i:i + n] for i in range(len(text) - n + 1))

    n1 = get_ngrams(t1)
    n2 = get_ngrams(t2)
    if not n1 or not n2:
        return 0.0
    return len(n1 & n2) / len(n1 | n2)


# ==============================
# NEW FEATURES
# ==============================

def token_sort_similarity(t1, t2):
    """
    Sort tokens alphabetically then compute char-ngram overlap.
    Catches paraphrases that reorder words:
      'the cat sat on the mat' vs 'the mat sat on the cat'
    Regular Jaccard misses this because sets are identical anyway,
    but char-ngram on the sorted string gives a different signal
    when the texts differ in more complex ways.
    """
    sorted1 = " ".join(sorted(t1.lower().split()))
    sorted2 = " ".join(sorted(t2.lower().split()))
    return char_ngram_similarity(sorted1, sorted2)


@lru_cache(maxsize=64_000)
def _synonyms(word):
    """Cached WordNet synonym lookup — avoids re-querying the same word."""
    syns = {word}
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            syns.add(lemma.name().replace("_", " "))
    return frozenset(syns)


def synonym_jaccard_similarity(t1, t2):
    """
    Jaccard on synonym-expanded token sets.
    Catches 'automobile' vs 'car', 'big' vs 'large', etc.
    Critical for semantic plagiarism where the writer swaps
    words with their synonyms.
    """
    words1 = t1.lower().split()
    words2 = t2.lower().split()

    if not words1 or not words2:
        return 0.0

    expanded1 = set()
    for w in words1:
        expanded1.update(_synonyms(w))

    expanded2 = set()
    for w in words2:
        expanded2.update(_synonyms(w))

    union = expanded1 | expanded2
    if not union:
        return 0.0
    return len(expanded1 & expanded2) / len(union)


def edit_distance_similarity(t1, t2):
    """
    Token-level normalised Levenshtein similarity.
    Catches near-duplicate sentences that differ by
    a few insertions / deletions / substitutions.
    O(m*n) in token counts — fast for sentence-length inputs.
    """
    words1 = t1.split()
    words2 = t2.split()

    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0

    m, n = len(words1), len(words2)
    # Single-row DP (memory-efficient)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[j] = prev[j - 1]
            else:
                dp[j] = 1 + min(prev[j], dp[j - 1], prev[j - 1])

    return 1.0 - dp[n] / max(m, n)


def lcs_similarity(t1, t2):
    """
    Token-level Longest Common Subsequence ratio (F1-style).
    Unlike Jaccard, LCS respects token order, so
    'I love NLP' and 'NLP I love' score differently.
    Complement to token_sort_similarity.
    """
    words1 = t1.split()
    words2 = t2.split()

    if not words1 or not words2:
        return 0.0

    m, n = len(words1), len(words2)
    # O(m*n) DP — fine for sentence-length texts
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if words1[i - 1] == words2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs_len = dp[m][n]
    # 2*LCS / (len1 + len2)  →  1.0 when identical, 0.0 when disjoint
    return (2 * lcs_len) / (m + n)


# ==============================
# SENTENCE-LEVEL SIM (inference only)
# ==============================

def sentence_level_similarity(text1, text2):
    """Used at inference time, not during training (too slow for 30k pairs)."""
    s1 = sent_tokenize(text1)
    s2 = sent_tokenize(text2)
    if not s1 or not s2:
        return 0.0
    emb1 = normalize(embed_batch(s1))
    emb2 = normalize(embed_batch(s2))
    sim_matrix = np.dot(emb1, emb2.T)
    return float((np.mean(np.max(sim_matrix, axis=1)) +
                  np.mean(np.max(sim_matrix, axis=0))) / 2)


# ==============================
# FEATURE MATRIX  (training)
# ==============================

def build_feature_matrix(df, tfidf_vectorizer=None):
    t1_clean = df["text1"].apply(lambda x: preprocess_text(x, "tfidf")).tolist()
    t2_clean = df["text2"].apply(lambda x: preprocess_text(x, "tfidf")).tolist()
    t1_raw   = df["text1"].apply(lambda x: preprocess_text(x, "embed")).tolist()
    t2_raw   = df["text2"].apply(lambda x: preprocess_text(x, "embed")).tolist()

    n = len(df)

    # ── TF-IDF cosine ──────────────────────────────────────────────
    tfidf_vectorizer, matrix = build_tfidf_matrix(
        t1_clean + t2_clean, tfidf_vectorizer
    )
    tfidf_sims = [
        tfidf_pair_similarity(matrix[i], matrix[n + i]) for i in range(n)
    ]

    # ── Document embedding cosine ───────────────────────────────────
    emb1 = normalize(embed_batch(t1_raw))
    emb2 = normalize(embed_batch(t2_raw))
    embed_sims = np.sum(emb1 * emb2, axis=1)

    # ── Original surface features ───────────────────────────────────
    len_sims     = [length_similarity(t1_clean[i],    t2_clean[i]) for i in range(n)]
    jaccard_sims = [jaccard_similarity(t1_clean[i],   t2_clean[i]) for i in range(n)]
    char_sims    = [char_ngram_similarity(t1_clean[i], t2_clean[i]) for i in range(n)]

    # ── NEW features ────────────────────────────────────────────────
    token_sort_sims  = [token_sort_similarity(t1_clean[i],    t2_clean[i]) for i in range(n)]
    synonym_sims     = [synonym_jaccard_similarity(t1_clean[i], t2_clean[i]) for i in range(n)]
    edit_dist_sims   = [edit_distance_similarity(t1_clean[i],  t2_clean[i]) for i in range(n)]
    lcs_sims         = [lcs_similarity(t1_clean[i],            t2_clean[i]) for i in range(n)]

    features_df = pd.DataFrame({
        "tfidf_cosine_sim":   tfidf_sims,
        "embed_cosine_sim":   embed_sims,
        "length_sim":         len_sims,
        "jaccard_sim":        jaccard_sims,
        "char_ngram_sim":     char_sims,
        "token_sort_sim":     token_sort_sims,      # NEW
        "synonym_jaccard_sim": synonym_sims,        # NEW
        "edit_distance_sim":  edit_dist_sims,       # NEW
        "lcs_sim":            lcs_sims,             # NEW
    })

    return features_df, tfidf_vectorizer


# ==============================
# HIGHLIGHT MATCHES  (improved)
# ==============================

def highlight_similar_sentences(text1, text2, threshold=0.70):
    """
    Bipartite matching instead of greedy argmax.

    Old approach: for each sentence in text1 pick the best in text2
    → the same text2 sentence can be matched many times (duplicate highlights).

    New approach: Hungarian algorithm (linear_sum_assignment) gives a
    one-to-one optimal assignment, then we keep only pairs above threshold.
    Remaining unmatched sentences from the longer text are checked
    individually so we don't silently drop valid matches.
    """
    s1 = sent_tokenize(text1)
    s2 = sent_tokenize(text2)

    if not s1 or not s2:
        return []

    emb1 = normalize(embed_batch(s1))
    emb2 = normalize(embed_batch(s2))
    sim_matrix = np.dot(emb1, emb2.T)   # shape: (len(s1), len(s2))

    matches = []
    used1   = set()
    used2   = set()

    # ── Optimal one-to-one matching on the square subproblem ────────
    min_len = min(len(s1), len(s2))
    row_ind, col_ind = linear_sum_assignment(-sim_matrix[:min_len, :min_len])

    for i, j in zip(row_ind, col_ind):
        score = float(sim_matrix[i, j])
        if score >= threshold:
            matches.append({
                "text1_sentence": s1[i],
                "text2_sentence": s2[j],
                "similarity":     round(score, 4),
            })
            used1.add(i)
            used2.add(j)

    # ── Remaining sentences in the longer text ───────────────────────
    # If text1 is longer, its extra sentences might still match something.
    for i in range(len(s1)):
        if i in used1:
            continue
        best_j = int(np.argmax(sim_matrix[i]))
        score  = float(sim_matrix[i, best_j])
        if score >= threshold and best_j not in used2:
            matches.append({
                "text1_sentence": s1[i],
                "text2_sentence": s2[best_j],
                "similarity":     round(score, 4),
            })
            used1.add(i)
            used2.add(best_j)

    matches.sort(key=lambda x: x["similarity"], reverse=True)
    return matches



