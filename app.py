import re
import math
import difflib

import numpy as np
import streamlit as st
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# KONFIGURASI
# ============================================================

SBERT_MODEL_NAME = ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
T5_MODEL_PATH = "dreasss/mongabay-t5-summarization"

SELECTION_RATIO = 0.50

GENERATION_CONFIG = {
    "max_new_tokens": 200,
    "min_length": 80,
    "num_beams": 6,
    "length_penalty": 1.0,
    "early_stopping": True,
    "no_repeat_ngram_size": 3,
    "repetition_penalty": 2.0,
}


NORMALIZATION_DICT = {
    "gk": "tidak",
    "ga": "tidak",
    "gak": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",
    "yg": "yang",
    "dgn": "dengan",
    "utk": "untuk",
    "krn": "karena",
    "karna": "karena",
    "dr": "dari",
    "dlm": "dalam",
    "sdh": "sudah",
    "blm": "belum",
    "jd": "jadi",
    "jgn": "jangan",
    "org": "orang",
    "pd": "pada",
    "tsb": "tersebut",
    "dpt": "dapat",
    "bgt": "banget",
    "aja": "saja",
    "kalo": "kalau",
    "tp": "tapi",
    "sm": "sama",
}


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource(show_spinner=False)
def load_sbert():
    return SentenceTransformer(SBERT_MODEL_NAME)

@st.cache_resource(show_spinner=False)
def load_t5(model_path):

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path
    )

    model.config.forced_bos_token_id = None

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device


# ============================================================
# PREPROCESSING
# ============================================================

def text_cleaning(text: str) -> str:

    if not isinstance(text, str):
        return ""

    text = text.lower().strip()

    text = re.sub(
        r"http\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"<.*?>",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


def normalize_text(text: str) -> str:

    if not isinstance(text, str):
        return ""

    tokens = text.split()

    tokens = [
        NORMALIZATION_DICT.get(tok, tok)
        for tok in tokens
    ]

    return " ".join(tokens)


def segment_sentences(text: str) -> list[str]:

    if not isinstance(text, str) or not text.strip():
        return []

    sentences = [
        s.strip()
        for s in re.split(
            r"(?<=[.!?])(?!\d)\s*",
            text
        )
        if s.strip()
    ]

    return sentences


def preprocess_article(text: str) -> tuple[str, list[str]]:

    cleaned = text_cleaning(text)

    normalized = normalize_text(cleaned)

    sentences = segment_sentences(normalized)

    return normalized, sentences


# ============================================================
# SBERT + COSINE SIMILARITY
# ============================================================

def calculate_representativeness(
    sentences,
    sbert_model
):

    if not sentences:
        return None, [], []

    embeddings = sbert_model.encode(
        sentences,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    sim_matrix = cosine_similarity(
        embeddings
    )

    n_total = len(sentences)

    if n_total > 1:

        avg_scores = (
            sim_matrix.sum(axis=1) - 1
        ) / (n_total - 1)

    else:

        avg_scores = np.ones(
            n_total,
            dtype=float
        )

    return (
        embeddings,
        sim_matrix,
        avg_scores.tolist()
    )


def select_candidate_sentences(
    sentences,
    scores,
    ratio=SELECTION_RATIO
):

    if not sentences or not scores:
        return ""

    k = max(
        1,
        math.ceil(
            len(sentences) * ratio
        )
    )

    top_indices = np.argsort(
        np.asarray(
            scores,
            dtype=float
        )
    )[-k:]

    # Pertahankan urutan kalimat asli
    top_indices = np.sort(
        top_indices
    )

    selected = [
        sentences[i]
        for i in top_indices
    ]

    return " ".join(selected)


# ============================================================
# GENERASI T5
# ============================================================

def generate_summary(
    text,
    tokenizer,
    model,
    device
):

    if not isinstance(text, str) or not text.strip():
        return ""

    input_text = "ringkas: " + text

    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    input_ids = inputs.input_ids.to(
        device
    )

    attention_mask = inputs.attention_mask.to(
        device
    )

    with torch.no_grad():

        summary_ids = model.generate(

            input_ids=input_ids,

            attention_mask=attention_mask,

            max_new_tokens=GENERATION_CONFIG[
                "max_new_tokens"
            ],

            min_length=GENERATION_CONFIG[
                "min_length"
            ],

            num_beams=GENERATION_CONFIG[
                "num_beams"
            ],

            length_penalty=GENERATION_CONFIG[
                "length_penalty"
            ],

            early_stopping=GENERATION_CONFIG[
                "early_stopping"
            ],

            no_repeat_ngram_size=GENERATION_CONFIG[
                "no_repeat_ngram_size"
            ],

            repetition_penalty=GENERATION_CONFIG[
                "repetition_penalty"
            ],

            forced_eos_token_id=(
                tokenizer.eos_token_id
            ),
        )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True,
    ).strip()

    summary = re.sub(
        r"\b([a-zA-Z])\s+([a-zA-Z]{3,})\b",
        r"\1\2",
        summary,
    )

    source_words = set(
        re.findall(
            r"\b[a-z]+(?:[-'][a-z]+)?\b",
            text.lower()
        )
    )

    def auto_correct(match):

        word = match.group(0)

        word_lower = word.lower()

        if (
            len(word_lower) <= 3
            or word_lower in source_words
        ):
            return word

        matches = difflib.get_close_matches(
            word_lower,
            source_words,
            n=1,
            cutoff=0.8,
        )

        if matches:

            corrected = matches[0]

            if word.istitle():
                return corrected.title()

            elif word.isupper():
                return corrected.upper()

            return corrected

        return word

    summary = re.sub(
        r"\b[a-zA-Z]+\b",
        auto_correct,
        summary,
    )

    sentences = re.split(
        r"(?<=[.])\s+",
        summary,
    )

    seen = []

    seen_lower = set()

    for sentence in sentences:

        normalized_sentence = (
            sentence.lower().strip()
        )

        if (
            normalized_sentence
            and normalized_sentence
            not in seen_lower
        ):

            seen.append(sentence)

            seen_lower.add(
                normalized_sentence
            )

    summary = " ".join(seen)

    if summary:

        summary = (
            summary.rstrip(".")
            + "."
        )

    return summary


# ============================================================
# PIPELINE LENGKAP
# ============================================================

def summarize_article(
    text,
    sbert_model,
    tokenizer,
    t5_model,
    device,
    progress_callback=None,
):

    def update_progress(progress, message):

        if progress_callback is not None:
            progress_callback(progress, message)

    update_progress(0.35, "Melakukan preprocessing artikel...")

    normalized_text, sentences = (
        preprocess_article(text)
    )

    if len(sentences) == 0:

        raise ValueError(
            "Tidak ditemukan kalimat yang "
            "dapat diproses setelah preprocessing."
        )

    update_progress(
        0.50,
        "Menghitung representativeness dengan SBERT...",
    )

    embeddings, sim_matrix, scores = (
        calculate_representativeness(
            sentences,
            sbert_model,
        )
    )

    update_progress(0.70, "Memilih kandidat kalimat terbaik...")

    selected_candidates = (
        select_candidate_sentences(
            sentences,
            scores,
            ratio=SELECTION_RATIO,
        )
    )

    if not selected_candidates.strip():

        raise ValueError(
            "Kandidat kalimat kosong."
        )

    update_progress(0.80, "Menghasilkan ringkasan dengan model T5...")

    summary = generate_summary(
        selected_candidates,
        tokenizer,
        t5_model,
        device,
    )

    update_progress(1.0, "Proses selesai.")

    return {

        "normalized_text":
            normalized_text,

        "sentences":
            sentences,

        "sentence_embeddings":
            embeddings,

        "similarity_matrix":
            sim_matrix,

        "avg_scores":
            scores,

        "selected_candidates":
            selected_candidates,

        "summary":
            summary,
    }


# ============================================================
# STREAMLIT INTERFACE
# ============================================================

st.set_page_config(
    page_title=(
        "Hybrid SBERT + T5 "
        "Text Summarization"
    ),

    page_icon="📝",

    layout="wide",
)


st.title(
    "📝 Text Summarization"
)


st.caption(
    "Extractive candidate selection "
    "berbasis SBERT + generasi ringkasan "
    "menggunakan T5 Fine-Tuned"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Konfigurasi Model"
    )

    st.write(
        f"**SBERT:** "
        f"`{SBERT_MODEL_NAME}`"
    )

    st.write(
        f"**T5:** "
        f"`{T5_MODEL_PATH}`"
    )

    st.write(
        f"**Seleksi kandidat:** "
        f"`{SELECTION_RATIO * 100:.0f}%`"
    )

    st.divider()

    st.write(
        "**Parameter generasi T5**"
    )

    st.write(
        f"- min_length: "
        f"`{GENERATION_CONFIG['min_length']}`"
    )

    st.write(
        f"- max_new_tokens: "
        f"`{GENERATION_CONFIG['max_new_tokens']}`"
    )

    st.write(
        f"- num_beams: "
        f"`{GENERATION_CONFIG['num_beams']}`"
    )

    st.write(
        f"- length_penalty: "
        f"`{GENERATION_CONFIG['length_penalty']}`"
    )

    st.write(
        f"- early_stopping: "
        f"`{GENERATION_CONFIG['early_stopping']}`"
    )

    st.write(
        f"- no_repeat_ngram_size: "
        f"`{GENERATION_CONFIG['no_repeat_ngram_size']}`"
    )

    st.write(
        f"- repetition_penalty: "
        f"`{GENERATION_CONFIG['repetition_penalty']}`"
    )


# ============================================================
# INPUT ARTIKEL
# ============================================================

st.subheader(
    "Masukkan Artikel"
)

def reset_article():
    st.session_state.article_input = ""

article = st.text_area(
    "Teks artikel",
    height=350,
    placeholder=(
        "Tempel artikel berita "
        "yang ingin diringkas di sini..."
    ),
    key="article_input",
)


col1, col2 = st.columns([2, 1])

with col1:
    process_button = st.button(
        "Ringkas",
        type="primary",
        use_container_width=True,
    )

with col2:
    reset_button = st.button(
        "Reset",
        type="primary",
        use_container_width=True,
        on_click=reset_article,
    )
# ============================================================
# PROSES
# ============================================================

if process_button:

    if not article.strip():

        st.warning(
            "Silakan masukkan teks artikel "
            "terlebih dahulu."
        )

        st.stop()

    try:

        progress_bar = st.progress(
            0,
            text="Menyiapkan proses... 0%",
        )

        def update_process(progress, message):
            progress_percent = round(progress * 100)
            progress_bar.progress(
                progress,
                text=f"{message} {progress_percent}%",
            )

        # ====================================================
        # LOAD SBERT
        # ====================================================

        update_process(0.05, "Memuat model SBERT...")
        with st.spinner("Memuat model SBERT..."):
            sbert_model = load_sbert()

        update_process(0.25, "Model SBERT siap. Memuat model T5...")


        # ====================================================
        # LOAD T5
        # ====================================================

        with st.spinner("Memuat model T5 Fine-Tuned..."):
            tokenizer, t5_model, device = (
                load_t5(
                    T5_MODEL_PATH
                )
            )

        update_process(0.35, "Model T5 siap. Memproses artikel...")


        # ====================================================
        # PIPELINE
        # ====================================================

        with st.spinner("Memproses artikel dan menghasilkan ringkasan..."):
            result = summarize_article(

                article,

                sbert_model,

                tokenizer,

                t5_model,

                device,

                progress_callback=update_process,
            )

            progress_bar.empty()


        st.success(
            "Ringkasan berhasil dibuat."
        )


        # ====================================================
        # OUTPUT RINGKASAN
        # ====================================================

        st.subheader(
            "📌 Hasil Ringkasan"
        )


        st.text_area(

            "Ringkasan",

            value=result["summary"],

            height=220,
        )


        # ====================================================
        # INFORMASI PIPELINE
        # ====================================================

        st.subheader(
            "📊 Informasi Proses"
        )


        selected_count = max(

            1,

            math.ceil(

                len(result["sentences"])

                * SELECTION_RATIO
            ),
        )


        st.write(
            f"**Jumlah Kalimat Awal:** "
            f"{len(result['sentences'])}"
        )


        st.write(
            f"**Kandidat Terpilih:** "
            f"{selected_count}"
        )


        st.write(
            f"**Rasio Seleksi:** "
            f"{SELECTION_RATIO * 100:.0f}%"
        )


        # ====================================================
        # SELECTED CANDIDATES
        # ====================================================

        with st.expander(
            "🔎 Lihat Selected Candidates dari SBERT"
        ):

            st.write(
                result["selected_candidates"]
            )


        # ====================================================
        # SKOR REPRESENTATIVENESS
        # ====================================================

        with st.expander(
            "📈 Lihat skor representativeness "
            "setiap kalimat"
        ):

            score_rows = []

            for idx, (
                sentence,
                score
            ) in enumerate(

                zip(
                    result["sentences"],
                    result["avg_scores"],
                ),

                start=1,
            ):

                score_rows.append({

                    "No":
                        idx,

                    "Kalimat":
                        sentence,

                    "Average Similarity":
                        round(
                            float(score),
                            6,
                        ),
                })


            for row in score_rows:

                st.write(

                    f"**Kalimat {row['No']}**  \n"

                    f"{row['Kalimat']}  \n"

                    f"Average Similarity: "
                    f"`{row['Average Similarity']}`"
                )


        # ====================================================
        # PREPROCESSING
        # ====================================================

        with st.expander(
            "🧹 Lihat hasil preprocessing"
        ):

            st.text_area(

                "Teks setelah preprocessing",

                value=result[
                    "normalized_text"
                ],

                height=220,
            )


    # ========================================================
    # ERROR MODEL
    # ========================================================

    except FileNotFoundError as exc:

        st.error(
            "Model T5 tidak dapat ditemukan."
        )

        st.exception(exc)


    # ========================================================
    # ERROR LAINNYA
    # ========================================================

    except Exception as exc:

        st.error(
            "Terjadi kesalahan saat "
            "menjalankan pipeline."
        )

        st.exception(exc)


# ============================================================
# FOOTER
# ============================================================

st.divider()


st.caption(
    "Pipeline: Preprocessing → "
    "Segmentasi Kalimat → SBERT → "
    "Cosine Similarity → Seleksi Kandidat 50% → "
    "T5 Fine-Tuned → Ringkasan"
)