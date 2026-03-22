import streamlit as st
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Meeting Intelligence Assistant V2", layout="wide")

st.title("🎧 Meeting Intelligence Assistant (V2)")
st.write("Ask questions and instantly find answers from meetings with timestamps.")

# -----------------------------
# Load models (cached)
# -----------------------------
@st.cache_resource
def load_whisper():
    return WhisperModel("medium")  # 👈 upgraded model

@st.cache_resource
def load_embedder():
    return SentenceTransformer('all-MiniLM-L6-v2')

whisper_model = load_whisper()
embed_model = load_embedder()

# -----------------------------
# Text cleaning (accent-robust)
# -----------------------------
def clean_text(text):
    # Keep only English characters + basic punctuation
    text = re.sub(r'[^A-Za-z0-9.,?!\'"()\-: ]+', ' ', text)
    
    # Normalize spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("📂 Upload audio file", type=["mp3", "wav", "m4a"])

if uploaded_file:
    # Read audio bytes
    audio_bytes = uploaded_file.read()

    # Save file
    with open("temp_audio.mp3", "wb") as f:
        f.write(audio_bytes)

    st.info("⏳ Processing audio... Please wait.")

    # Show audio player
    st.audio(audio_bytes)

    # -----------------------------
    # Transcription (accent improved)
    # -----------------------------
    with st.spinner("Transcribing audio..."):
        segments, _ = whisper_model.transcribe(
            "temp_audio.mp3",
            language="en",
            condition_on_previous_text=False,
            beam_size=5,
            vad_filter=True
        )

    chunks = []
    for seg in segments:
        cleaned = clean_text(seg.text)
        if cleaned:
            chunks.append({
                "text": cleaned,
                "start": seg.start,
                "end": seg.end
            })

    st.success("✅ Transcription complete!")

    # -----------------------------
    # Embeddings + FAISS
    # -----------------------------
    texts = [c["text"] for c in chunks]

    with st.spinner("Generating embeddings..."):
        embeddings = embed_model.encode(texts)

    index = faiss.IndexFlatL2(len(embeddings[0]))
    index.add(np.array(embeddings))

    st.success("✅ Ready! Ask your question below.")

    # -----------------------------
    # Search function
    # -----------------------------
    def search(query, k=3):
        q_emb = embed_model.encode([query])
        D, I = index.search(np.array(q_emb), k)
        return [chunks[i] for i in I[0]]

    # -----------------------------
    # Query input
    # -----------------------------
    query = st.text_input("🔍 Ask a question about the meeting:")

    if query:
        results = search(query)

        # -----------------------------
        # AI-style answer
        # -----------------------------
        context = " ".join([r["text"] for r in results])

        answer = f"""
Based on the meeting discussion:

{context}
        """

        st.markdown("### 🤖 AI Answer")
        st.write(answer)

        # -----------------------------
        # Results with audio jump
        # -----------------------------
        st.markdown("### 📌 Relevant Moments")

        for i, r in enumerate(results):
            st.markdown(
                f"""
**⏱ {r['start']:.2f}s - {r['end']:.2f}s**  
{r['text']}
                """
            )

            if st.button(f"▶️ Play from {r['start']:.2f}s", key=i):
                st.audio(audio_bytes, start_time=int(r['start']))
