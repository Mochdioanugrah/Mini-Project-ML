# Klasifikasi Berita Hoax & Non-Hoax di Indonesia  
*Mini Project – Pembelajaran Mesin (Machine Learning, NLP, Deep Learning, Transformer)*

---

## URGENSI  
Perkembangan teknologi informasi membuat penyebaran berita sangat cepat melalui media sosial dan portal daring. Namun, kondisi ini mendorong meningkatnya **berita hoaks**, yang berdampak pada:

- Kesalahpahaman publik  
- Konflik sosial  
- Manipulasi opini politik/ekonomi  
- Kepanikan massal  

Validasi manual berita hoaks memakan banyak waktu dan sumber daya. Karena itu, penelitian ini bertujuan membangun sistem otomatis berbasis **Machine Learning (ML)** dan **Natural Language Processing (NLP)** untuk mendeteksi hoaks secara cepat, akurat, dan dapat diskalakan.

Pendekatan yang digunakan:
- **ML klasik** (Logistic Regression, SVM, Random Forest)  
- **Deep Learning** (MLP, BiLSTM)  
- **IndoBERT Transformer (Fine-Tuning & LoRA PEFT)**  
- **Explainable AI (LIME, SHAP, Attention, IG)**  

---

# Deskripsi Singkat  
Repository ini berisi pipeline lengkap:

- Scraping dataset (TurnBackHoax.id & Detik.com)
- Preprocessing teks lengkap
- Ekstraksi fitur: TF-IDF & Word2Vec
- Balancing data dengan SMOTE
- Seleksi fitur: Chi-Square, ANOVA F-test, MI
- Pelatihan model ML, DL, Transformer
- Evaluasi performa menyeluruh
- Analisis kesalahan & interpretabilitas model (XAI)

---

# Tujuan Penelitian  
- Membangun sistem otomatis untuk mengklasifikasikan berita hoaks.  
- Menganalisis pengaruh representasi fitur terhadap performa model.  
- Membandingkan model ML klasik, Deep Learning, dan Transformer.  
- Mengidentifikasi fitur/kata paling berpengaruh.  
- Meninjau penyebab akurasi model dan potensi data leakage.  

---

# Dataset  

## 1. TurnBackHoax.id (Berita Hoaks)  
Contoh data:

| Judul | URL | Konten |
|-------|---------|---------|
| [PENIPUAN] Tautan Pendaftaran PMO | https://turnbackhoax.id/... | Tautan palsu beredar melalui TikTok… |
| [SALAH] Rumah Kapolda Bali Digeruduk Massa | https://turnbackhoax.id/... | Klaim menyesatkan terkait video kerusuhan… |

Filtering label berdasarkan regex:  
`"HOAKS", "SALAH", "DISINFORMASI", "MISLEADING", "FALSE", "FABRICATED", "PENIPUAN"`

---

## 2. Detik.com (Non-Hoaks)
Contoh data:

| Judul | Link | Isi Berita |
|-------|---------|--------------|
| Hamas Tewaskan 33 Orang… | https://news.detik.com/... | Pembaruan konflik Gaza… |
| Polda Metro Luncurkan Layanan SIKAP | https://news.detik.com/... | Layanan digital pelaporan masyarakat… |

---

## Hasil Penggabungan Dataset  
Total: **4.491** artikel  
- **600** hoaks  
- **3.891** non-hoaks  

Contoh gabungan:

| Isi Berita | Label |
|------------|--------|
| Menaker Sebut Transisi Ekonomi Hijau… | non-hoaks |
| Karakter Naga Jawa… | non-hoaks |

---

# Preprocessing  
Tahapan:

1. Case folding  
2. Cleaning (hapus angka, simbol, tag)  
3. Tokenisasi  
4. Stopword removal  
5. Stemming  
6. Rejoin token → kalimat final  
7. Label encoding (hoaks = 1, non-hoaks = 0)

Contoh:

| Tahap | Output |
|-------|---------|
| Tokenisasi | ['tautan','pendaftaran','project'] |
| Stopword removal | ['akun','tiktok','info'] |
| Stemming | ['video','tampil','banjir'] |
| Final text | video tampil banjir besar jakarta |

---

# Feature Extraction  

## TF-IDF  
- Total fitur awal: **5000**
- Seleksi fitur: **2000 terbaik** melalui ANOVA F-test

Contoh bobot tertinggi:

| Dokumen | Kata | Bobot |
|---------|--------|--------|
| 1 | koperasi | 0.322 |
| 2 | anung | 0.300 |
| 3 | kapolda | 0.294 |

---

## Word2Vec  
- Model: **CBOW**  
- Dimensi vektor: **100**  
- Digunakan untuk model sekuens (BiLSTM)  

---

# Train–Test Split  

| Jenis | Jumlah Data | Non-Hoax | Hoax |
|--------|----------------|-------------|---------|
| Train | 3.584 | 86.83% | 13.17% |
| Test | 897 | 86.85% | 13.15% |

---

# Balancing Data (SMOTE)

| Representasi | Sebelum | Sesudah | Distribusi |
|--------------|--------------|--------------|----------------|
| TF-IDF | 3.584 | 6.224 | 0:3112, 1:3112 |
| Word2Vec | 3.584 | 6.224 | 0:3112, 1:3112 |

---

# Seleksi Fitur (TF-IDF)

| Metode | Fitur Awal | Fitur Terpilih | Skor |
|--------|--------------|----------------|--------|
| Chi-Square | 5000 | 2000 | 4.6558 |
| **ANOVA F-test** | 5000 | 2000 | **146.3059** |
| Mutual Information | 5000 | 2000 | 0.0161 |

ANOVA → **paling stabil & informatif**

---

# Model Machine Learning Klasik  

## TF-IDF
| Model | Akurasi | Precision | Recall | F1 |
|--------|----------|------------|-----------|--------|
| Logistic Regression | 0.9978 | 1.00 | 0.98–1.00 | 0.99–1.00 |
| **SVM** | **1.0000** | 1.00 | 1.00 | 1.00 |
| **Random Forest** | **1.0000** | 1.00 | 1.00 | 1.00 |

## Word2Vec
| Model | Akurasi | Precision | Recall | F1 |
|--------|----------|------------|-----------|--------|
| Logistic Regression | 0.9944 | 0.97–1.00 | 0.98–1.00 | 0.98–1.00 |
| SVM | 0.9955 | 0.98–1.00 | 0.98–1.00 | 0.99 |
| Random Forest | 0.9933 | 0.99–1.00 | 0.96–1.00 | 0.97–0.99 |

---

# Error Analysis  

Penyebab potensi akurasi tinggi:

- Kata unik yang sangat khas antar kelas  
- SMOTE bisa mendekati pola data uji  
- Seleksi fitur sebelum split → rawan **data leakage**  
- Struktur bahasa TurnBackHoax vs Detik sangat berbeda  

Solusi:

- Stratified k-fold cross validation  
- Pengujian dataset eksternal  
- Pemisahan pipeline preprocessing secara ketat  

---

# Model Deep Learning  

## 1. MLP + TF-IDF (Chi-Square)
- Akurasi: **99.89%**
- Salah prediksi: **1 dari 897**

| Pred / True | Non-Hoax | Hoax |
|-------------|------------|--------|
| Non-Hoax | 779 | 0 |
| Hoax | 1 | 117 |

---

## 2. BiLSTM + Word2Vec
- Akurasi: **100%**

| Pred / True | Non-Hoax | Hoax |
|-------------|------------|--------|
| Non-Hoax | 778 | 0 |
| Hoax | 0 | 574 |

---

# 3. Transformer — IndoBERT (Full Fine-Tuning)

| Kelas | Precision | Recall | F1 | Support |
|--------|------------|-----------|----------|------------|
| Non-Hoax | 1.00 | 1.00 | 1.00 | 778 |
| Hoax | 1.00 | 1.00 | 1.00 | 574 |
| **Total** | — | — | **1.00** | 1352 |

---

# 4. Transformer — IndoBERT + LoRA (PEFT)

- Akurasi identik dengan fine-tuning penuh  
- Latihan lebih ringan dan efisien  

| Kelas | Precision | Recall | F1 |
|--------|------------|-----------|----------|
| Non-Hoax | 1.00 | 1.00 | 1.00 |
| Hoax | 1.00 | 1.00 | 1.00 |

---

# Explainable AI (XAI)

Metode interpretasi:
- **LIME**  
- **SHAP**  
- **Attention Visualization**  
- **Integrated Gradients**  

Kata paling berpengaruh:
- “palsu”  
- “bukti”  
- “presiden”  
- “tanpa bukti”  

---

# Perbandingan Semua Model 

| Model                      | Representasi            | Akurasi   | Precision | Recall | F1-Score | Catatan Singkat |
|---------------------------|--------------------------|-----------|-----------|--------|----------|------------------|
| Logistic Regression       | TF-IDF                   | **0.9985** | 1.00      | 1.00   | 1.00     | Stabil, akurat |
| SVM (LinearSVC)           | TF-IDF                   | **1.0000** | 1.00      | 1.00   | 1.00     | Model klasik terbaik |
| Random Forest             | TF-IDF                   | **1.0000** | 1.00      | 1.00   | 1.00     | Sempurna |
| Logistic Regression       | Word2Vec                 | **0.9896** | 0.99      | 0.99   | 0.99     | Word2Vec kurang optimal |
| SVM                       | Word2Vec                 | **0.9970** | 1.00      | 1.00   | 1.00     | Hampir setara TF-IDF |
| Random Forest             | Word2Vec                 | **0.9815** | 0.98–0.99 | 0.97–0.99 | 0.98 | Performa terendah |
| MLP (Dense NN)            | TF-IDF                   | **1.0000** | 1.00      | 0.99–1.00 | 1.00 | Deep learning sederhana, hasil sempurna |
| BiLSTM                    | Word2Vec                 | **1.0000** | 1.00      | 1.00   | 1.00     | Word2Vec + LSTM sangat efektif |
| IndoBERT (Full Fine-Tuning) | Transformer     | **1.0000** | 1.00 | 1.00 | 1.00 | Model terbaik; memahami konteks mendalam |
| IndoBERT + LoRA (PEFT) | Transformer + LoRA | **1.0000** | 1.00 | 1.00 | 1.00 | Lebih efisien; hasil setara full fine-tuning |

---

# Insight Utama  
- TF-IDF paling efektif untuk ML klasik  
- Word2Vec sangat baik untuk model sekuensial  
- IndoBERT unggul jauh karena memahami konteks  
- LoRA mempermudah fine-tuning tanpa menurunkan akurasi  
- Sumber dataset berbeda mempermudah pemisahan kelas  

---

# Kesimpulan  
- **IndoBERT** adalah model terbaik (100% akurasi, tidak ada kesalahan).  
- **BiLSTM** menjadi deep learning terbaik kedua.  
- **SVM + TF-IDF** adalah model klasik paling konsisten.  
- Representasi fitur memengaruhi performa secara signifikan.  
- Sistem dapat dikembangkan menjadi API pendeteksi hoaks realtime.  

