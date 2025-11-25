# Klasifikasi Berita Hoax & Non-Hoax di Indonesia  
*Mini Project – Pembelajaran Mesin*

Repositori ini berisi implementasi proyek machine learning untuk mendeteksi berita **hoax** dan **non-hoax** menggunakan pendekatan *Natural Language Processing (NLP)*. Proyek mencakup proses scraping data, preprocessing teks, ekstraksi fitur, balancing data dengan SMOTE, seleksi fitur, pelatihan model, evaluasi, hingga analisis hasil.
Tujuannya adalah membangun model klasifikasi teks yang akurat dengan membandingkan berbagai metode representasi fitur dan algoritma machine learning.

---

##  Tujuan Penelitian  
- Mengembangkan model otomatis untuk klasifikasi berita hoax dan non-hoax.  
- Menganalisis perbedaan performa model dengan dua representasi fitur: **TF-IDF** dan **Word2Vec**.  
- Membandingkan berbagai model machine learning klasik.  
- Mengidentifikasi fitur/kata yang paling berpengaruh dalam klasifikasi.  

---

## Dataset  

Dataset diambil dari dua sumber resmi:

### 🔹 Sumber Data  
| Sumber | Keterangan |
|--------|------------|
| **TurnBackHoax.id** | Scraping berita hoax yang telah diverifikasi MAFINDO. |
| **Detik.com** | Scraping berita valid (non-hoax). |

### 🔹 Ringkasan Dataset Gabungan  
- **Total data:** 4.491 berita  
  - **600** hoaks  
  - **3.891** non-hoaks  
- **File:** `dataset_hoax_vs_nonhoax_final_filtered.csv`

### 🔹 Contoh Data  
| Isi Berita | Label |
|------------|--------|
| Menaker Sebut Transisi Ekonomi Hijau Momentum ... | non-hoaks |
| akun tiktok info baru pada kamis unggah foto ... | hoaks |

---

## Representasi Fitur  

### 1. TF-IDF  
- 5.000 fitur awal  
- Seleksi fitur → 2.000 fitur (menggunakan ANOVA F-test)

#### Hasil Seleksi Fitur  
| Metode | Fitur Awal | Fitur Terpilih | Rata-rata Skor |
|--------|------------|----------------|----------------|
| Chi-Square | 5000 | 2000 | 4.6558 |
| ANOVA F-test | 5000 | 2000 | 146.3059 |
| Mutual Information | 5000 | 2000 | 0.0161 |

---

### 2. Word2Vec  
- Model **CBOW**  
- Dimensi vektor **100-dim**  
- Representasi berbasis makna (semantic embedding)

---

##  Model Machine Learning yang Diuji  

| Model | Jenis | Kelebihan | Kekurangan |
|-------|--------|------------|------------|
| Logistic Regression | Linear | Cepat dan sederhana | Kurang baik untuk pola non-linear |
| SVM (LinearSVC) | Margin-based | Akurasi tinggi untuk teks | Lambat pada dataset besar |
| Random Forest | Ensemble | Tahan noise dan kuat | Interpretasi lebih sulit |

---

## Hasil Eksperimen  

### 🔹 A. TF-IDF + Seleksi Fitur  
| Model | Akurasi | Precision | Recall | F1-Score |
|--------|---------|------------|---------|-----------|
| Logistic Regression | 0.9978 | 1.00 | 0.98–1.00 | 0.99–1.00 |
| SVM | **1.0000** | 1.00 | 1.00 | 1.00 |
| Random Forest | **1.0000** | 1.00 | 1.00 | 1.00 |

---

### 🔹 B. Word2Vec (100-D)  
| Model | Akurasi | Precision | Recall | F1-Score |
|--------|---------|------------|---------|-----------|
| Logistic Regression | 0.9944 | 0.97–1.00 | 0.98–1.00 | 0.98–1.00 |
| SVM | **0.9955** | 0.98–1.00 | 0.98–1.00 | 0.99 |
| Random Forest | 0.9933 | 0.99–1.00 | 0.96–1.00 | 0.97–0.99 |

---

###  Perbandingan Akurasi TF-IDF vs Word2Vec  
| Model | TF-IDF (%) | Word2Vec (%) |
|--------|------------|--------------|
| Logistic Regression | 99.78 | 99.44 |
| SVM | **100.00** | 99.55 |
| Random Forest | **100.00** | 99.33 |

---

##  Insight  

1. **SVM + TF-IDF** memberikan hasil paling sempurna (100%).  
2. Representasi **TF-IDF** lebih efektif dibanding Word2Vec pada dataset pendek.  
3. Dataset relatif mudah dipisahkan karena gaya bahasa hoax & non-hoax berbeda signifikan.  
4. Performa sangat tinggi perlu diuji ulang dengan **cross-validation** karena berpotensi ada data leakage.  
5. SMOTE berhasil mengatasi ketidakseimbangan kelas.

---

##  Kesimpulan  

- **SVM + TF-IDF (2000 fitur ANOVA)** adalah kombinasi terbaik.  
- TF-IDF memberikan performa paling stabil pada klasifikasi berita.  
- Word2Vec tetap menghasilkan akurasi sangat tinggi namun sedikit di bawah TF-IDF.  
- Model perlu diuji pada dataset eksternal untuk validasi lebih kuat.  
- Sistem dapat dikembangkan menjadi API deteksi hoax atau integrasi ke aplikasi web.

---
