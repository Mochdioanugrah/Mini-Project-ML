# 📰 Klasifikasi Berita Hoax & Non-Hoax di Indonesia  
*Mini Project – Pembelajaran Mesin (Machine Learning & NLP)*

## URGENSI  
Perkembangan teknologi informasi membuat penyebaran berita menjadi sangat cepat melalui media sosial dan portal daring. Namun, hal ini juga memicu meningkatnya **berita hoaks**, yang dapat menyebabkan:

- Kesalahpahaman publik  
- Konflik sosial  
- Manipulasi opini  
- Kepanikan massal  

Validasi manual berita hoaks memakan waktu besar. Karena itu, sistem otomatis berbasis **Machine Learning (ML)** dan **Natural Language Processing (NLP)** diperlukan untuk mendeteksi hoaks secara cepat dan akurat. Penelitian ini membangun sistem klasifikasi hoaks berbahasa Indonesia menggunakan **ML klasik**, **Deep Learning**, dan **Transformer (IndoBERT)**.

---

## Deskripsi Singkat  
Repository ini berisi keseluruhan pipeline deteksi hoaks:

- Web scraping dataset (TurnBackHoax & Detik)  
- Preprocessing teks lengkap  
- Feature extraction: **TF-IDF**, **Word2Vec**  
- Balancing data menggunakan SMOTE  
- Feature selection (Chi-Square, ANOVA F-test, Mutual Information)  
- Model ML klasik: Logistic Regression, SVM, Random Forest  
- Model Deep Learning: **MLP**, **BiLSTM**  
- Model Transformer: **IndoBERT (Fine-Tuned)**  
- Evaluasi lengkap: accuracy, precision, recall, F1  
- Error analysis dan insight  

---

## Tujuan Penelitian  
- Membangun model otomatis untuk mendeteksi berita hoaks.  
- Menganalisis pengaruh representasi teks terhadap performa model.  
- Perbandingan metode ML klasik vs Deep Learning vs Transformer.  
- Mengidentifikasi fitur/kata paling berpengaruh.  
- Melakukan analisis kesalahan untuk mencegah overfitting/data leakage.

---

# Dataset  

## 1. TurnBackHoax.id (Berita Hoaks)
Contoh:

| Judul | URL | Konten |
|-------|---------|---------|
| PENIPUAN – Tautan Pendaftaran PMO | https://turnbackhoax.id/... | Tautan palsu beredar melalui TikTok… |
| SALAH – Rumah Kapolda Bali Digeruduk Massa | https://turnbackhoax.id/... | Klaim menyesatkan terkait video kerusuhan… |

---

## 2. Detik.com (Berita Non-Hoaks)
Contoh:

| Judul | Link | Isi Berita |
|-------|---------|--------------|
| Hamas Tewaskan 33 Orang… | https://news.detik.com/... | Pembaruan konflik Gaza… |
| Polda Metro Luncurkan Layanan SIKAP | https://news.detik.com/... | Layanan digital pelaporan masyarakat… |

---

## 🔹 Hasil Penggabungan Dataset  
Total data: **4.491**  
- Hoaks: **600**  
- Non-hoaks: **3.891**

Contoh dataset gabungan:

| Isi Berita | Label |
|------------|--------|
| Menaker Sebut Transisi Ekonomi Hijau… | non-hoaks |
| Karakter Naga Jawa… | non-hoaks |

---

# 🧹 Preprocessing  

Tahapan preprocessing:

1. **Case Folding**  
2. **Cleaning (hapus simbol, angka)**  
3. **Tokenisasi**  
4. **Stopword Removal**  
5. **Stemming**  
6. **Rejoin token → kalimat final**  
7. **Label encoding (hoaks = 1, non-hoaks = 0)**

Contoh hasil:

| Tahapan | Output |
|---------|---------|
| Tokenisasi | ['tautan','pendaftaran','project',...] |
| Stopword removal | ['akun','tiktok','info','terbaru',...] |
| Stemming | ['video','tampil','banjir','besar'] |
| Final text | video tampil banjir besar jakarta |

---

# Feature Extraction  

## TF-IDF  
Menangkap kata-kata unik per dokumen.  
5.000 fitur awal → diseleksi menjadi 2.000 fitur.

Top kata TF-IDF:

| Dokumen | Kata | Bobot |
|---------|--------|--------|
| 1 | koperasi | 0.322 |
| 2 | anung | 0.300 |
| 3 | kapolda | 0.294 |

---

## Word2Vec  
- Model: **CBOW**  
- Dimensi: **100-dim**  
- Panjang sequence: **120**  
- Representasi semantik lebih dalam dibanding TF-IDF  

---

# Train–Test Split  

| Jenis | Jumlah | Non-Hoax | Hoax |
|--------|-----------|------------|--------|
| Train | 3.584 | 86.83% | 13.17% |
| Test | 897 | 86.85% | 13.15% |

---

# Balancing Data (SMOTE)

| Representasi | Sebelum | Sesudah | Distribusi |
|--------------|--------------|--------------|----------------|
| TF-IDF | 3584 | 6224 | 0:3112, 1:3112 |
| Word2Vec | 3584 | 6224 | 0:3112, 1:3112 |

---

# Seleksi Fitur (TF-IDF)

| Metode | Fitur Awal | Fitur Terpilih | Skor |
|--------|--------------|----------------|--------|
| Chi-Square | 5000 | 2000 | 4.6558 |
| ANOVA F-test | 5000 | 2000 | **146.3059** |
| Mutual Information | 5000 | 2000 | 0.0161 |

ANOVA F-test → **metode terbaik**

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
Beberapa potensi penyebab akurasi terlalu tinggi:

- Kata-kata sangat khas tiap kelas → mudah dipisahkan  
- SMOTE mungkin menciptakan pola yang mirip data uji  
- Seleksi fitur dilakukan sebelum split (potensi data leakage)  
- Dataset cukup homogen  

Pencegahan:

- **Stratified k-Fold Cross Validation**  
- Pengujian pada dataset baru/eksternal  

---

# Model Deep Learning  

## 1. MLP + TF-IDF (Chi-Square)
- Akurasi: **99.89%**
- Salah prediksi: **1 dari 897**

Confusion Matrix:

| Pred / True | Non-Hoax | Hoax |
|-------------|------------|--------|
| Non-Hoax | 779 | 0 |
| Hoax | 1 | 117 |

---

## 2. BiLSTM + Word2Vec
- Akurasi: **100%**
- Tidak ada kesalahan

| Pred / True | Non-Hoax | Hoax |
|-------------|------------|--------|
| Non-Hoax | 778 | 0 |
| Hoax | 0 | 574 |

---

# 3. Transformer Model — IndoBERT (Fine-Tuned)

| Kelas | Precision | Recall | F1 | Support |
|--------|------------|-----------|----------|------------|
| Non-Hoax | 1.00 | 1.00 | 1.00 | 778 |
| Hoax | 1.00 | 1.00 | 1.00 | 574 |
| **Total** | — | — | **1.00** | 1352 |

**IndoBERT adalah model terbaik di seluruh percobaan.**

---

# Perbandingan Semua Model

| Model               | Representasi | Akurasi    | Precision | Recall    | F1-Score | Catatan Singkat                                 |
| ------------------- | ------------ | ---------- | --------- | --------- | -------- | ----------------------------------------------- |
| Logistic Regression | TF-IDF       | **0.9985** | 1.00      | 1.00      | 1.00     | Sangat akurat, model klasik yang stabil         |
| SVM (LinearSVC)     | TF-IDF       | **1.0000** | 1.00      | 1.00      | 1.00     | Performa sempurna; model klasik terbaik         |
| Random Forest       | TF-IDF       | **1.0000** | 1.00      | 1.00      | 1.00     | Sempurna; cocok untuk dataset linear/non-linear |
| Logistic Regression | Word2Vec     | **0.9896** | 0.99      | 0.99      | 0.99     | Menurun sedikit; Word2Vec kurang optimal        |
| SVM                 | Word2Vec     | **0.9970** | 1.00      | 1.00      | 1.00     | Hampir sempurna; robust di Word2Vec             |
| Random Forest       | Word2Vec     | **0.9815** | 0.98–0.99 | 0.97–0.99 | 0.98     | Performa terendah, masih sangat baik            |
| MLP                 | TF-IDF       | **1.0000** | 1.00      | 0.99–1.00 | 1.00     | Deep learning sederhana, hasil sempurna         |
| BiLSTM              | Word2Vec     | **1.0000** | 1.00      | 1.00      | 1.00     | Sempurna; Word2Vec + LSTM bekerja sangat baik   |
| IndoBERT            | Transformer  | **1.0000** | 1.00      | 1.00      | 1.00     | Terbaik dari semua model; memahami konteks      |

---

# Insight Utama  
- TF-IDF kuat untuk model klasik  
- Word2Vec sangat cocok untuk model sekuensial (BiLSTM)  
- IndoBERT mengalahkan semua model karena memahami konteks lebih dalam  
- SMOTE + feature selection meningkatkan performa signifikan  
- Dataset cenderung mudah dipisahkan karena sumber data jelas berbeda  

---

# Kesimpulan  
- **IndoBERT adalah model terbaik dan paling akurat (100%)**  
- **BiLSTM** menduduki posisi kedua dengan performa sempurna  
- **SVM + TF-IDF** tetap juara dalam kategori model klasik  
- Representasi fitur sangat memengaruhi hasil  
- Sistem ini dapat dikembangkan sebagai API pendeteksi hoaks  

---
