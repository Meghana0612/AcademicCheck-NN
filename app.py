import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from itertools import combinations
from pypdf import PdfReader

from src.predictor import PlagiarismPredictor


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AcademicCheck NN",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================
# LIGHT, CLEAN THEME
# =========================
st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(180deg, #f8fbff 0%, #f4f7ff 45%, #fffafc 100%);
            color: #0f172a;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
            border-right: 1px solid #e2e8f0;
        }

        [data-testid="stSidebar"] * {
            color: #0f172a !important;
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #0f172a;
            margin-bottom: 0.25rem;
        }

        .sub-title {
            font-size: 1rem;
            color: #475569;
            margin-bottom: 1rem;
        }

        .card {
            background: rgba(255, 255, 255, 0.98);
            padding: 18px;
            border-radius: 18px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
            margin-bottom: 16px;
        }

        .metric-card {
            background: white;
            border-radius: 18px;
            padding: 16px;
            border: 1px solid #e2e8f0;
            box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
        }

        .about-box {
            background: linear-gradient(180deg, #eff6ff 0%, #eef2ff 100%);
            border: 1px solid #bfdbfe;
            padding: 14px;
            border-radius: 14px;
            color: #0f172a;
            line-height: 1.6;
            font-size: 0.92rem;
        }

        .stButton > button {
            background: linear-gradient(90deg, #2563eb, #7c3aed);
            color: white;
            border-radius: 12px;
            border: none;
            padding: 0.7rem 1.05rem;
            font-weight: 700;
            box-shadow: 0 10px 20px rgba(37, 99, 235, 0.20);
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(37, 99, 235, 0.24);
        }

        .stDownloadButton > button {
            background: linear-gradient(90deg, #16a34a, #22c55e);
            color: white;
            border-radius: 12px;
            border: none;
            padding: 0.7rem 1.05rem;
            font-weight: 700;
            box-shadow: 0 10px 20px rgba(34, 197, 94, 0.18);
        }

        textarea {
            background: #ffffff !important;
            color: #0f172a !important;
        }

        label, p, span, div {
            color: #0f172a;
        }

        /* ── Selectbox: closed control ─────────────────────────────── */
        [data-baseweb="select"] {
            background-color: #ffffff !important;
        }

        [data-baseweb="select"] > div,
        [data-baseweb="select"] > div > div {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1px solid #cbd5e1 !important;
        }

        [data-baseweb="select"] span,
        [data-baseweb="select"] input,
        [data-baseweb="select"] div {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

        [data-baseweb="select"] svg {
            fill: #0f172a !important;
        }

        /* ── Selectbox: dropdown popup container ───────────────────── */
        [data-baseweb="popover"],
        [data-baseweb="popover"] > div,
        [data-baseweb="popover"] > div > div {
            background-color: #ffffff !important;
            border: 1px solid #e2e8f0 !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 24px rgba(15, 23, 42, 0.10) !important;
        }

        /* ── Selectbox: menu / list wrapper ────────────────────────── */
        [data-baseweb="menu"],
        [data-baseweb="menu"] > ul,
        ul[role="listbox"],
        [role="listbox"] {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

        /* ── Selectbox: individual options ─────────────────────────── */
        [role="option"],
        [data-baseweb="menu"] [role="option"],
        ul[role="listbox"] li {
            background-color: #ffffff !important;
            color: #0f172a !important;
            padding: 10px 14px !important;
        }

        [role="option"]:hover,
        [data-baseweb="menu"] [role="option"]:hover,
        ul[role="listbox"] li:hover {
            background-color: #dbeafe !important;
            color: #1d4ed8 !important;
            cursor: pointer;
        }

        [aria-selected="true"],
        [data-baseweb="menu"] [aria-selected="true"],
        ul[role="listbox"] [aria-selected="true"] {
            background-color: #eff6ff !important;
            color: #1d4ed8 !important;
            font-weight: 600 !important;
        }

        /* ── Catch-all: any child inside popover must be light ──────── */
        [data-baseweb="popover"] * {
            background-color: #ffffff !important;
            color: #0f172a !important;
        }

        [data-baseweb="popover"] [role="option"]:hover {
            background-color: #dbeafe !important;
            color: #1d4ed8 !important;
        }

        [data-baseweb="popover"] [aria-selected="true"] {
            background-color: #eff6ff !important;
            color: #1d4ed8 !important;
        }

        /* ── Dataframes / tables ────────────────────────────────────── */
        [data-testid="stDataFrame"] {
            border-radius: 14px;
            overflow: hidden;
        }

        th, td {
            color: #0f172a !important;
        }

        /* ── Tabs ───────────────────────────────────────────────────── */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #ffffff;
            border-radius: 999px;
            padding: 8px 14px;
            border: 1px solid #e2e8f0;
            font-weight: 600;
            color: #334155;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(90deg, #eff6ff, #f5f3ff) !important;
            border: 1px solid #93c5fd !important;
            color: #1d4ed8 !important;
        }

        details > summary {
            color: #0f172a !important;
            font-weight: 700;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================
# MODEL LOAD
# =========================
@st.cache_resource
def load_engine():
    return PlagiarismPredictor(model_dir="models")


predictor = load_engine()


# =========================
# SESSION STATE
# =========================
if "batch_ready" not in st.session_state:
    st.session_state.batch_ready = False

if "batch_data" not in st.session_state:
    st.session_state.batch_data = {}

if "selected_metric" not in st.session_state:
    st.session_state.selected_metric = "probability"

if "show_about" not in st.session_state:
    st.session_state.show_about = False


# =========================
# TEXT EXTRACTION
# =========================
def extract_text(uploaded_file):
    if uploaded_file.name.lower().endswith(".pdf"):
        try:
            reader = PdfReader(uploaded_file)
            pages = []
            for page in reader.pages:
                pages.append(page.extract_text() or "")
            return "\n".join(pages).strip()
        except Exception as e:
            st.error(f"Could not read PDF {uploaded_file.name}: {e}")
            return ""
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore").strip()


# =========================
# REPORT TEXT
# =========================
def make_report_text(result, name1, name2):
    lines = []
    lines.append("PLAGIARISM DETECTION REPORT")
    lines.append("=" * 50)
    lines.append(f"Document 1: {name1}")
    lines.append(f"Document 2: {name2}")
    lines.append(f"Verdict   : {result.get('verdict')}")
    lines.append(f"Probability: {result.get('probability')}%")
    lines.append(f"Threshold : {result.get('threshold')}")
    lines.append("")
    lines.append("SIMILARITY SCORES")
    lines.append("-" * 50)

    keys = [
        "tfidf_similarity",
        "embedding_similarity",
        "length_similarity",
        "jaccard_similarity",
        "char_ngram_similarity",
        "token_sort_similarity",
        "synonym_jaccard_similarity",
        "edit_distance_similarity",
        "lcs_similarity",
    ]
    for k in keys:
        if k in result:
            lines.append(f"{k}: {result[k]}")

    lines.append("")
    lines.append("MATCHED SENTENCES")
    lines.append("-" * 50)
    if result.get("matched_sentences"):
        for m in result["matched_sentences"]:
            lines.append(f"Text1: {m['text1_sentence']}")
            lines.append(f"Text2: {m['text2_sentence']}")
            lines.append(f"Similarity: {m['similarity']}")
            lines.append("")
    else:
        lines.append("No strong matches found.")

    lines.append("")
    lines.append("EXPLAINABLE AI")
    lines.append("-" * 50)
    shap_vals = result.get("shap_values", {}).get("values", {})
    if shap_vals:
        for k, v in shap_vals.items():
            lines.append(f"SHAP {k}: {v}")

    lime_vals = result.get("lime_weights", [])
    if lime_vals:
        lines.append("")
        for item in lime_vals:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                lines.append(f"LIME {item[0]}: {item[1]}")

    return "\n".join(lines)


# =========================
# DISPLAY HELPERS
# =========================
def render_result_header(result):
    verdict   = result.get("verdict", "Unknown")
    prob      = result.get("probability", 0)
    threshold = result.get("threshold", 0.5)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.9rem;color:#64748b;font-weight:700">Verdict</div>
                <div style="font-size:1.75rem;font-weight:800;color:#0f172a;margin-top:6px">{verdict}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.9rem;color:#64748b;font-weight:700">Probability</div>
                <div style="font-size:1.75rem;font-weight:800;color:#7c3aed;margin-top:6px">{prob}%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.9rem;color:#64748b;font-weight:700">Threshold</div>
                <div style="font-size:1.75rem;font-weight:800;color:#16a34a;margin-top:6px">{threshold}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_similarity_cards(result):
    items = [
        ("TF-IDF",          result.get("tfidf_similarity", 0),            "#2563eb"),
        ("Embedding",        result.get("embedding_similarity", 0),        "#ea580c"),
        ("Length",           result.get("length_similarity", 0),           "#16a34a"),
        ("Jaccard",          result.get("jaccard_similarity", 0),          "#db2777"),
        ("Char N-Gram",      result.get("char_ngram_similarity", 0),       "#7c3aed"),
        ("Token Sort",       result.get("token_sort_similarity", 0),       "#0891b2"),
        ("Synonym Jaccard",  result.get("synonym_jaccard_similarity", 0),  "#ca8a04"),
        ("Edit Distance",    result.get("edit_distance_similarity", 0),    "#dc2626"),
        ("LCS",              result.get("lcs_similarity", 0),              "#0f766e"),
    ]

    rows = [items[i:i + 3] for i in range(0, len(items), 3)]
    for row in rows:
        cols = st.columns(3)
        for col, (label, value, color) in zip(cols, row):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div style="font-size:0.82rem;color:#64748b;font-weight:700;text-transform:uppercase">{label}</div>
                        <div style="font-size:2rem;font-weight:800;color:{color};margin-top:8px">{value:.2f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def render_explainability(result):
    st.markdown("### Explainable AI")

    shap_values  = result.get("shap_values", {}).get("values", {})
    lime_weights = result.get("lime_weights", [])

    if shap_values:
        st.markdown("#### SHAP-style Feature Contributions")
        shap_df = pd.DataFrame(
            [{"Feature": k, "Contribution": v} for k, v in shap_values.items()]
        ).sort_values(by="Contribution", key=lambda s: s.abs(), ascending=False)

        fig1 = px.bar(
            shap_df,
            x="Contribution",
            y="Feature",
            orientation="h",
            color="Contribution",
            color_continuous_scale="RdBu",
            title="Feature Contributions",
            text="Contribution",
            template="plotly_white",
        )
        fig1.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=45, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0f172a"),
        )
        fig1.update_xaxes(tickfont=dict(color="#0f172a"))
        fig1.update_yaxes(tickfont=dict(color="#0f172a"))
        st.plotly_chart(fig1, use_container_width=True)

    if lime_weights:
        st.markdown("#### LIME-style Local Weights")
        lime_df = pd.DataFrame(lime_weights, columns=["Feature", "Weight"])
        fig2 = px.bar(
            lime_df,
            x="Weight",
            y="Feature",
            orientation="h",
            color="Weight",
            color_continuous_scale="Viridis",
            title="Local Feature Weights",
            text="Weight",
            template="plotly_white",
        )
        fig2.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=45, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0f172a"),
        )
        fig2.update_xaxes(tickfont=dict(color="#0f172a"))
        fig2.update_yaxes(tickfont=dict(color="#0f172a"))
        st.plotly_chart(fig2, use_container_width=True)

    with st.expander("Show raw explanation values"):
        st.json({"shap_values": shap_values, "lime_weights": lime_weights})


def highlight_matches(matches):
    html = ""
    for m in matches:
        sim = float(m["similarity"])
        if sim > 0.85:
            color = "#16a34a"
            bg    = "#f0fdf4"
        elif sim > 0.65:
            color = "#eab308"
            bg    = "#fffbeb"
        else:
            color = "#ef4444"
            bg    = "#fef2f2"

        html += f"""
        <div style="background:{bg};padding:14px;border-left:6px solid {color};
        border-radius:12px;margin-bottom:12px;border:1px solid #e2e8f0">
        <div style="font-weight:700;color:#0f172a">Text1</div>
        <div style="color:#0f172a">{m['text1_sentence']}</div>
        <br>
        <div style="font-weight:700;color:#0f172a">Text2</div>
        <div style="color:#0f172a">{m['text2_sentence']}</div>
        <br>
        <b style="color:#0f172a">Similarity:</b> <span style="color:#0f172a">{round(sim, 4)}</span>
        </div>
        """
    return html


def show_result(result, name1, name2):
    st.markdown(f"## {name1} vs {name2}")

    render_result_header(result)

    st.markdown("### Similarity Breakdown")
    render_similarity_cards(result)

    st.markdown("### Matched Sentences")
    if result.get("matched_sentences"):
        st.markdown(highlight_matches(result["matched_sentences"]), unsafe_allow_html=True)
    else:
        st.info("No strong matches found.")

    render_explainability(result)

    report_text = make_report_text(result, name1, name2)
    st.download_button(
        "Download Report",
        data=report_text,
        file_name=f"{name1}_vs_{name2}.txt",
        mime="text/plain",
    )


# =========================
# PAIRWISE MATRIX HELPERS
# =========================
def build_pairwise_matrix(names, pair_results, metric_key):
    n      = len(names)
    matrix = np.zeros((n, n), dtype=float)

    for i in range(n):
        matrix[i, i] = 100.0

    idx = {name: i for i, name in enumerate(names)}

    for rec in pair_results:
        a   = rec["file1"]
        b   = rec["file2"]
        val = float(rec["result"].get(metric_key, rec["result"].get("probability", 0.0)))
        i   = idx[a]
        j   = idx[b]
        matrix[i, j] = val
        matrix[j, i] = val

    return pd.DataFrame(matrix, index=names, columns=names)


# =========================
# SIDEBAR
# =========================
st.sidebar.title("AcademicCheck NN")
mode = st.sidebar.radio("Mode", ["Text Input", "Multi File Upload"])
st.sidebar.markdown("---")

if st.sidebar.button("About"):
    st.session_state.show_about = not st.session_state.show_about

if st.session_state.show_about:
    st.sidebar.markdown(
        """
        <div class="about-box">
        <b>Overview of Our Project</b><br><br>
        AcademicCheck NN is a semantic plagiarism detection system that compares documents
        using a CrossEncoder-based model and multiple similarity measures.<br><br>
        It supports:
        <br>• single text pair analysis
        <br>• multi-file N×N comparison
        <br>• PDF and TXT uploads
        <br>• explainable AI outputs
        <br>• similarity heatmaps
        <br>• matched sentence highlighting<br><br>
        The system combines transformer-based semantic understanding with traditional NLP
        features such as TF-IDF, Jaccard similarity, character n-grams, token sort,
        synonym overlap, edit distance, and LCS.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.sidebar.markdown("---")


# =========================
# MAIN TITLE
# =========================
st.markdown(
    '<div class="main-title">🎓 AcademicCheck Multi-Format Analysis</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-title">Semantic plagiarism detection with explainable AI and batch comparison.</div>',
    unsafe_allow_html=True,
)

if not predictor.is_ready():
    st.error("Model not loaded. Run training first.")
    st.stop()


# =========================
# TEXT MODE
# =========================
if mode == "Text Input":
    c1, c2 = st.columns(2)

    with c1:
        t1 = st.text_area(
            "Text 1",
            height=260,
            placeholder="Paste or type the first document here...",
            key="text1_input",
        )
    with c2:
        t2 = st.text_area(
            "Text 2",
            height=260,
            placeholder="Paste or type the second document here...",
            key="text2_input",
        )

    if st.button("Check"):
        if not t1.strip() or not t2.strip():
            st.error("Both texts must be non-empty.")
        else:
            with st.spinner("Analyzing..."):
                res = predictor.predict(t1, t2)
            show_result(res, "Text1", "Text2")


# =========================
# MULTI FILE MODE
# =========================
else:
    st.markdown("### Upload Multiple Files")
    uploaded_files = st.file_uploader(
        "Upload .txt or .pdf files",
        accept_multiple_files=True,
        type=["txt", "pdf"],
        key="batch_upload",
    )

    if uploaded_files and len(uploaded_files) >= 2:
        total_comp = len(uploaded_files) * (len(uploaded_files) - 1) // 2
        st.info(
            f"{len(uploaded_files)} files uploaded. Total pairwise comparisons: {total_comp}"
        )

        if st.button("Compare All Files"):
            names = [f.name for f in uploaded_files]

            with st.spinner("Extracting text from documents..."):
                texts = [extract_text(f) for f in uploaded_files]

            valid = [(n, t) for n, t in zip(names, texts) if len(t.strip()) > 10]
            if len(valid) < 2:
                st.error("Not enough valid text found in uploaded files.")
                st.stop()

            names = [x[0] for x in valid]
            texts = [x[1] for x in valid]
            n     = len(names)

            results     = []
            total_pairs = n * (n - 1) // 2
            progress    = st.progress(0)
            step        = 0

            with st.spinner(f"Analyzing {total_pairs} relationships..."):
                for i, j in combinations(range(n), 2):
                    res = predictor.predict(texts[i], texts[j])
                    results.append(
                        {
                            "file1":  names[i],
                            "file2":  names[j],
                            "result": res,
                        }
                    )
                    step += 1
                    progress.progress(step / total_pairs)

            st.session_state.batch_data  = {
                "names":   names,
                "texts":   texts,
                "results": results,
            }
            st.session_state.batch_ready = True

    elif uploaded_files:
        st.info("Upload at least 2 files.")

    if st.session_state.batch_ready and st.session_state.batch_data:
        data    = st.session_state.batch_data
        names   = data["names"]
        results = data["results"]

        metric_keys = [
            "probability",
            "tfidf_similarity",
            "embedding_similarity",
            "length_similarity",
            "jaccard_similarity",
            "char_ngram_similarity",
            "token_sort_similarity",
            "synonym_jaccard_similarity",
            "edit_distance_similarity",
            "lcs_similarity",
        ]

        st.markdown("## Similarity Heatmap")
        st.markdown(
            "<div style='font-size:0.95rem;font-weight:700;color:#334155;margin-bottom:0.35rem'>Select Matrix Metric</div>",
            unsafe_allow_html=True,
        )
        selected_metric = st.selectbox(
            "",
            metric_keys,
            index=metric_keys.index(st.session_state.selected_metric)
            if st.session_state.selected_metric in metric_keys
            else 0,
            format_func=lambda x: x.replace("_", " ").title(),
            key="selected_metric_box",
            label_visibility="collapsed",
        )
        st.session_state.selected_metric = selected_metric

        heat_df = build_pairwise_matrix(names, results, selected_metric)
        fig = px.imshow(
            heat_df,
            text_auto=".1f",
            color_continuous_scale="RdYlBu_r",
            aspect="auto",
            labels=dict(color="Score"),
            template="plotly_white",
        )
        fig.update_layout(
            height=650,
            margin=dict(l=10, r=10, t=40, b=10),
            plot_bgcolor="white",
            paper_bgcolor="white",
            font=dict(color="#0f172a"),
        )
        fig.update_xaxes(tickfont=dict(color="#0f172a"))
        fig.update_yaxes(tickfont=dict(color="#0f172a"))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(heat_df, use_container_width=True, hide_index=False)

        st.markdown("## High Risk Matches")
        threshold  = getattr(predictor, "best_threshold", 0.5) * 100
        match_rows = []
        for rec in results:
            if rec["result"]["probability"] >= threshold:
                match_rows.append(
                    {
                        "Doc A":         rec["file1"],
                        "Doc B":         rec["file2"],
                        "Probability %": f"{rec['result']['probability']:.1f}",
                        "Verdict":       rec["result"]["verdict"],
                    }
                )

        if match_rows:
            st.dataframe(pd.DataFrame(match_rows), use_container_width=True, hide_index=True)
        else:
            st.success("No significant plagiarism detected.")

        st.markdown("## Pairwise Detailed Results")
        st.markdown(
            "<div style='font-size:0.95rem;font-weight:700;color:#334155;margin-bottom:0.35rem'>Select a comparison</div>",
            unsafe_allow_html=True,
        )
        pair_labels = [f"{r['file1']} vs {r['file2']}" for r in results]
        chosen      = st.selectbox(
            "",
            pair_labels,
            key="pair_select_box",
            label_visibility="collapsed",
        )
        chosen_idx = pair_labels.index(chosen)
        chosen_rec = results[chosen_idx]
        show_result(chosen_rec["result"], chosen_rec["file1"], chosen_rec["file2"])

        combined_text = []
        for rec in results:
            r = rec["result"]
            combined_text.append(
                f"{rec['file1']} vs {rec['file2']} -> {r['verdict']} ({r['probability']}%)"
            )

        st.download_button(
            "Download All Results",
            data="\n".join(combined_text),
            file_name="all_results.txt",
            mime="text/plain",
        )



        