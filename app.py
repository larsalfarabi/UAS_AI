import streamlit as st
import joblib
import pickle
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ====== Konfigurasi Halaman ======
st.set_page_config(
    page_title="HoaxRadar — Deteksi Berita Politik Indonesia",
    page_icon=":material/radar:",
    layout="centered"
)

# ====== Custom CSS ======
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Header utama */
.hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2.8rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0.2rem;
    line-height: 1.2;
}

.hero-subtitle {
    color: #94a3b8;
    text-align: center;
    font-size: 0.95rem;
    margin-bottom: 2rem;
    letter-spacing: 0.05em;
}

/* Badge pipeline */
.pipeline-badge {
    display: inline-block;
    background: rgba(167, 139, 250, 0.15);
    border: 1px solid rgba(167, 139, 250, 0.3);
    color: #a78bfa;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 2px;
}

.pipeline-container {
    text-align: center;
    margin-bottom: 2rem;
}

/* Card statistik */
.stat-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    backdrop-filter: blur(10px);
}

.stat-number {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.8rem;
    font-weight: 700;
    color: #60a5fa;
}

.stat-label {
    color: #94a3b8;
    font-size: 0.8rem;
    margin-top: 4px;
}

/* Hasil hoax */
.result-hoax {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(220,38,38,0.05));
    border: 1px solid rgba(239,68,68,0.4);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.result-valid {
    background: linear-gradient(135deg, rgba(52,211,153,0.15), rgba(16,185,129,0.05));
    border: 1px solid rgba(52,211,153,0.4);
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    margin: 1rem 0;
}

.result-label-hoax {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f87171;
}

.result-label-valid {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #34d399;
}

.result-desc {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.5rem;
}

/* Text area */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 12px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}

/* Button */
.stButton button {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}

.stButton button:hover {
    opacity: 0.85 !important;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.1) !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #475569;
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(255,255,255,0.06);
}
</style>
""", unsafe_allow_html=True)

# ====== Load Model ======
@st.cache_resource
def load_model():
    try:
        with open('models/hoax_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

@st.cache_resource
def load_preprocessor():
    factory_stemmer = StemmerFactory()
    stemmer = factory_stemmer.create_stemmer()
    factory_stopword = StopWordRemoverFactory()
    stopword_remover = factory_stopword.create_stop_word_remover()
    return stemmer, stopword_remover

model, vectorizer = load_model()
stemmer, stopword_remover = load_preprocessor()

# ====== Fungsi Preprocessing ======
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text

# ====== HERO SECTION ======
st.markdown('<div class="hero-title">HoaxRadar</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Sistem Deteksi Berita Hoax Politik Indonesia berbasis Machine Learning</div>', unsafe_allow_html=True)

st.markdown("""
<div class="pipeline-container">
    <span class="pipeline-badge">Case Folding</span>
    <span style="color:#475569; margin:0 4px;">→</span>
    <span class="pipeline-badge">Cleaning</span>
    <span style="color:#475569; margin:0 4px;">→</span>
    <span class="pipeline-badge">Stopword Removal</span>
    <span style="color:#475569; margin:0 4px;">→</span>
    <span class="pipeline-badge">Sastrawi Stemmer</span>
    <span style="color:#475569; margin:0 4px;">→</span>
    <span class="pipeline-badge">TF-IDF</span>
    <span style="color:#475569; margin:0 4px;">→</span>
    <span class="pipeline-badge">SVM</span>
</div>
""", unsafe_allow_html=True)

# ====== STATISTIK ======
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">99.59%</div>
        <div class="stat-label">Akurasi Model SVM</div>
    </div>""", unsafe_allow_html=True)
with col2:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">20.780</div>
        <div class="stat-label">Total Data Training</div>
    </div>""", unsafe_allow_html=True)
with col3:
    st.markdown("""
    <div class="stat-card">
        <div class="stat-number">4</div>
        <div class="stat-label">Sumber Dataset</div>
    </div>""", unsafe_allow_html=True)

st.write("")
st.divider()

# ====== INPUT SECTION ======
st.markdown("### Analisis Berita")
st.markdown('<p style="color:#94a3b8; font-size:0.9rem;">Masukkan judul atau isi berita politik yang ingin diverifikasi.</p>', unsafe_allow_html=True)

news_text = st.text_area(
    label="Teks Berita",
    height=180,
    placeholder="Contoh: Pemerintah mengumumkan kebijakan baru terkait pemilu 2024...",
    label_visibility="collapsed"
)

analyze = st.button("Analisis Sekarang", use_container_width=True)

# ====== HASIL ======
if analyze:
    if not news_text.strip():
        st.warning("Harap masukkan teks berita terlebih dahulu.")
    elif model is None:
        st.error("Model tidak tersedia. Periksa file model di folder models/")
    else:
        with st.spinner("Menganalisis dengan pipeline NLP..."):
            clean_text = preprocess_text(news_text)
            text_tfidf = vectorizer.transform([clean_text])
            prediction = model.predict(text_tfidf)[0]

        st.write("")

        if prediction == 1:
            st.markdown("""
            <div class="result-hoax">
                <div class="result-label-hoax">TERINDIKASI HOAX</div>
                <div class="result-desc">Berita ini memiliki pola yang serupa dengan konten hoax dalam dataset pelatihan.</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-valid">
                <div class="result-label-valid">BERITA VALID</div>
                <div class="result-desc">Berita ini memiliki pola yang serupa dengan berita faktual dari sumber kredibel.</div>
            </div>""", unsafe_allow_html=True)

        # Detail
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Jumlah Kata (Asli)", f"{len(news_text.split())} kata")
        with col_b:
            st.metric("Token Setelah NLP", f"{len(clean_text.split())} token")

        with st.expander("Lihat detail preprocessing"):
            st.markdown("**Teks setelah pipeline NLP:**")
            st.code(clean_text, language=None)

        st.caption("Disclaimer: Hasil ini bersifat prediktif dan tidak menggantikan verifikasi dari CekFakta.com, TurnBackHoax.id, atau Kominfo.")

# ====== FOOTER ======
st.markdown("""
<div class="footer">
    UAS Artificial Intelligence · Universitas Cakrawala · Genap 2025/2026<br>
    Dataset: CNN Indonesia · Kompas · Tempo · TurnBackHoax.id
</div>
""", unsafe_allow_html=True)
