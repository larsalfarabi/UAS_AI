import streamlit as st
import joblib
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

# ====== Konfigurasi Halaman ======
st.set_page_config(
    page_title="Deteksi Hoax Berita Politik Indonesia",
    page_icon=":material/fact_check:",
    layout="centered"
)

# ====== Load Model ======
@st.cache_resource
def load_model():
    model = joblib.load('models/hoax_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

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

# ====== Header ======
st.title("Deteksi Hoax Berita Politik Indonesia")
st.caption("Pipeline NLP: Case Folding → Cleaning → Stopword Removal → Sastrawi Stemmer → TF-IDF → SVM")
st.divider()

st.markdown("""
Sistem ini menganalisis teks berita politik menggunakan pipeline NLP berbahasa Indonesia
untuk mendeteksi apakah berita tersebut terindikasi **Hoax** atau **Valid**.

Dataset: CNN Indonesia, Kompas, Tempo (berita valid) + TurnBackHoax.id (berita hoax)
Total data training: **20.780 berita** dengan akurasi model **99.59%**
""")

st.write("")

# ====== Input ======
with st.form("hoax_form"):
    st.subheader(":material/article: Input Teks Berita")
    news_text = st.text_area(
        "Masukkan judul atau isi berita yang ingin dianalisis:",
        height=200,
        placeholder="Contoh: Pemerintah mengumumkan kebijakan baru terkait pemilu 2024..."
    )
    submitted = st.form_submit_button(
        ":material/search: Analisis Sekarang",
        use_container_width=True,
        type="primary"
    )

# ====== Hasil Prediksi ======
if submitted:
    if not news_text.strip():
        st.warning("Harap masukkan teks berita terlebih dahulu.")
    else:
        try:
            with st.spinner("Memproses teks dengan pipeline NLP..."):
                clean_text = preprocess_text(news_text)
                text_tfidf = vectorizer.transform([clean_text])
                prediction = model.predict(text_tfidf)[0]

            st.divider()
            st.subheader(":material/insights: Hasil Analisis")

            col1, col2, col3 = st.columns(3)

            with col1:
                if prediction == 1:
                    st.error("TERINDIKASI HOAX")
                    st.metric("Status", "Hoax")
                else:
                    st.success("BERITA VALID")
                    st.metric("Status", "Valid")

            with col2:
                st.metric("Jumlah Kata", f"{len(news_text.split())} kata")

            with col3:
                st.metric("Token Setelah NLP", f"{len(clean_text.split())} token")

            with st.expander("Lihat detail hasil preprocessing"):
                st.markdown("**Teks asli:**")
                st.write(news_text)
                st.markdown("**Setelah pipeline NLP (Case Folding → Cleaning → Stopword → Stemming):**")
                st.code(clean_text)

            st.caption("Disclaimer: Hasil analisis bersifat prediktif berdasarkan model ML dan tidak menggantikan verifikasi faktual dari sumber terpercaya seperti Kominfo, TurnBackHoax.id, atau CekFakta.com.")

        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
            st.info("Pastikan folder models/ berisi hoax_model.pkl dan tfidf_vectorizer.pkl")

st.write("")
st.divider()
st.caption("UAS Artificial Intelligence | Universitas Cakrawala | Genap 2025/2026")
