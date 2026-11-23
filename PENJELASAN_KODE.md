# Penjelasan Kode per Bagian

File ini menjelaskan **apa yang dilakukan setiap bagian kode** di
`preprocessing_ulasan_ecommerce.ipynb`, alasan langkah tersebut dibutuhkan, dan
cara mengatasi error yang umum muncul di Google Colab.

## Ringkasan Alur

```text
Dataset HuggingFace
        |
        v
DataFrame pandas
        |
        v
Case folding -> Cleaning -> Tokenisasi -> Stopword removal -> Stemming
        |
        v
Tabel 10 contoh + statistik sebelum vs sesudah
```

Urutan tersebut penting. Case folding dan cleaning dilakukan pada level teks,
sedangkan stopword removal dan stemming dilakukan pada level token (kata).

---

## 1. Persiapan Library

```python
%pip install -q datasets Sastrawi

import datasets
import Sastrawi
import pandas
import matplotlib

print("datasets   :", datasets.__version__)
print("pandas     :", pandas.__version__)
print("matplotlib :", matplotlib.__version__)
print("Sastrawi   : siap")
```

Penjelasan:

- `%pip` memasang package ke **environment kernel yang sedang aktif**. Di Colab,
  ini lebih aman daripada `!pip` karena memastikan package masuk ke Python yang
  dipakai notebook.
- `-q` berarti *quiet*, yaitu menyembunyikan output instalasi yang panjang.
- `datasets` dipakai untuk mengunduh dataset dari HuggingFace.
- `Sastrawi` dipakai untuk stopword removal dan stemming bahasa Indonesia.
- `pandas` dipakai untuk mengolah data dalam bentuk tabel.
- `matplotlib` dipakai untuk membuat grafik batang.
- Baris `print` memastikan semua library benar-benar bisa diimpor sebelum lanjut.

Jika Colab menampilkan **"You must restart the runtime"**, lakukan
**Runtime -> Restart session**, lalu jalankan ulang dari sel ini. Ini normal
ketika Colab memperbarui library yang sudah terpasang.

---

## 2. Memuat Dataset

```python
dataset = load_dataset(
    "dipawidia/ecommerce-product-reviews-sentiment",
    split="train"
)
df = dataset.to_pandas()
```

Penjelasan:

- `load_dataset(...)` mengunduh dataset dari HuggingFace Hub. Dataset akan
  disimpan di cache, sehingga pemanggilan berikutnya tidak mengunduh ulang.
- `split="train"` langsung mengambil bagian data latih. Dataset ini hanya punya
  split `train`.
- `to_pandas()` mengubah objek `Dataset` menjadi **pandas DataFrame**. DataFrame
  berbentuk tabel: setiap baris adalah satu ulasan, setiap kolom adalah atribut.
- Hasilnya: `df` berisi 11.606 baris dan 3 kolom: `review`, `translate`,
  `sentimen`.

Kolom yang dipakai dalam tugas ini adalah `review` (teks ulasan) dan `sentimen`
(label). Kolom `translate` tidak dipakai karena hanya berisi terjemahan.

---

## 3. Informasi dan Distribusi Dataset

```python
print("Jumlah data :", len(df))
print("Nama kolom  :", df.columns.tolist())
```

Penjelasan:

- `len(df)` menghitung jumlah baris (ulasan).
- `df.columns.tolist()` menampilkan nama-nama kolom dalam bentuk list.

```python
jumlah = df["sentimen"].value_counts().sort_index()
persen = df["sentimen"].value_counts(normalize=True).sort_index() * 100
```

Penjelasan:

- `df["sentimen"]` mengambil kolom label.
- `value_counts()` menghitung banyaknya data untuk setiap nilai label.
- `sort_index()` mengurutkan hasil berdasarkan kode label: `0` lalu `1`.
- `normalize=True` mengubah hasil menjadi proporsi (0 sampai 1).
- Dikali `100` supaya menjadi persentase.

```python
ringkasan = pd.DataFrame({
    "Kode": jumlah.index,
    "Sentimen": ["Negatif" if kode == 0 else "Positif" for kode in jumlah.index],
    "Jumlah": jumlah.values,
    "Persentase (%)": persen.round(2).values
})
ringkasan
```

Penjelasan:

- `jumlah.index` berisi kode label, yaitu `0` dan `1`.
- List comprehension mengubah kode menjadi nama yang mudah dibaca.
- `jumlah.values` dan `persen.values` mengambil angka tanpa index.
- `round(2)` membulatkan persentase menjadi dua angka di belakang koma.
- Baris terakhir (`ringkasan`) menampilkan tabel di notebook.

```python
ax = jumlah.plot(kind="bar", color=["#e74c3c", "#2ecc71"])
ax.set_title("Distribusi Sentimen Ulasan")
ax.set_xlabel("Sentimen (0 = negatif, 1 = positif)")
ax.set_ylabel("Jumlah ulasan")
ax.set_xticklabels(["Negatif (0)", "Positif (1)"], rotation=0)
```

Penjelasan:

- `plot(kind="bar")` membuat grafik batang dari jumlah tiap label.
- `color` mengatur warna batang: merah untuk negatif, hijau untuk positif.
- `set_title`, `set_xlabel`, dan `set_ylabel` memberi judul dan label sumbu.
- `set_xticklabels` mengganti angka `0` dan `1` dengan nama label.
- `rotation=0` membuat label sumbu X tetap mendatar.

```python
for i, nilai in enumerate(jumlah.values):
    ax.text(i, nilai + 80, str(nilai), ha="center")

plt.tight_layout()
plt.show()
```

Penjelasan:

- Loop menulis angka jumlah di atas setiap batang.
- `i` adalah posisi batang, `nilai` adalah tingginya.
- `nilai + 80` memberi sedikit jarak agar angka tidak menempel pada batang.
- `plt.tight_layout()` merapikan jarak elemen grafik.
- `plt.show()` menampilkan grafik.

Catatan: Grafik ini membantu melihat apakah data seimbang. Pada dataset ini
jumlahnya **5.803 negatif dan 5.803 positif**, jadi seimbang 50:50.

---

## 4. Preprocessing Teks

Preprocessing bertujuan membuat teks lebih seragam dan mengurangi noise. Setiap
tahap dibuat sebagai fungsi terpisah agar mudah diuji dan dipakai ulang.

### 4.1 Case Folding

```python
def case_folding(teks):
    # Mengubah seluruh huruf di teks menjadi huruf kecil
    return teks.lower()
```

Penjelasan:

- `teks.lower()` mengubah semua huruf menjadi huruf kecil.
- Tanpa case folding, `"Bagus"` dan `"bagus"` dianggap dua kata berbeda.
- Case folding dilakukan **sebelum** cleaning agar pola regex `[^a-z\s]` dapat
  membuang karakter selain huruf kecil dan spasi.

### 4.2 Cleaning

```python
def cleaning(teks):
    # Menghapus URL, angka, tanda baca, dan karakter khusus
    teks = re.sub(r"http\S+|www\.\S+", " ", teks)  # hapus URL
    teks = re.sub(r"\d+", " ", teks)               # hapus angka
    teks = re.sub(r"[^a-z\s]", " ", teks)          # sisakan huruf a-z dan spasi
    teks = re.sub(r"\s+", " ", teks)               # rapikan spasi berlebih
    return teks.strip()
```

Penjelasan pola regex:

| Pola | Arti |
|---|---|
| `r"http\S+\|www\.\S+"` | Mencari URL yang diawali `http` atau `www.` |
| `r"\d+"` | Mencari satu atau lebih angka |
| `r"[^a-z\s]"` | Mencari karakter selain huruf `a-z` dan spasi |
| `r"\s+"` | Mencari satu atau lebih spasi/newline |

- Setiap pola diganti dengan spasi `" "`.
- `re.sub` mengganti **semua** kecocokan, bukan hanya yang pertama.
- `[^a-z\s]` membuang tanda baca dan karakter khusus seperti `!`, `?`, `👍`,
  `@`, dan `#`.
- `\s+` menjadi satu spasi agar tidak ada spasi ganda.
- `.strip()` membuang spasi di awal dan akhir teks.
- Hasil cleaning masih berupa **satu string**, belum berupa daftar kata.

### 4.3 Tokenisasi

```python
def tokenisasi(teks):
    # Memecah teks menjadi daftar kata berdasarkan spasi
    return teks.split()
```

Penjelasan:

- `split()` memecah string berdasarkan spasi.
- Hasilnya berupa list token, misalnya
  `["suara", "mic", "puas", "pesan"]`.
- Tokenisasi diperlukan karena stopword removal dan stemming bekerja per kata.

### 4.4 Stopword Removal dengan Sastrawi

```python
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

stopword_sastrawi = set(StopWordRemoverFactory().get_stop_words())

def hapus_stopword(tokens):
    # Membuang token yang termasuk stopword Sastrawi
    return [kata for kata in tokens if kata not in stopword_sastrawi]
```

Penjelasan:

- `StopWordRemoverFactory()` membuat objek factory dari Sastrawi.
- `get_stop_words()` mengambil daftar stopword bahasa Indonesia.
- `set(...)` mengubah list menjadi set. Pengecekan `kata not in ...` pada set
  lebih cepat daripada pada list.
- List comprehension mempertahankan kata yang **tidak** ada di daftar stopword.
- Contoh stopword: `dan`, `di`, `yang`, `saya`, `tidak`.

Catatan penting: Sastrawi juga memasukkan kata negasi seperti `tidak` dan
`bukan` sebagai stopword. Untuk analisis sentimen, kata negasi sebaiknya
dipertahankan karena dapat membalik makna. Pada tugas ini kita mengikuti
instruksi memakai daftar stopword Sastrawi standar.

### 4.5 Stemming dengan Sastrawi dan Cache

```python
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stemmer = StemmerFactory().create_stemmer()
cache_stemming = {}

def stemming(tokens):
    # Mengubah token menjadi kata dasar menggunakan cache
    hasil = []

    for kata in tokens:
        if kata not in cache_stemming:
            cache_stemming[kata] = stemmer.stem(kata)
        hasil.append(cache_stemming[kata])

    return hasil
```

Penjelasan:

- `StemmerFactory().create_stemmer()` membuat objek stemmer Sastrawi.
- `stemmer.stem(kata)` mengubah kata berimbuhan menjadi kata dasar, misalnya
  `pengiriman` menjadi `kirim`.
- `cache_stemming` adalah **dictionary** yang menyimpan hasil per kata unik.
- Jika kata belum ada di cache, hasil stemming dihitung sekali lalu disimpan.
- Jika kata sudah ada, hasil diambil langsung dari cache tanpa menghitung ulang.
- `hasil.append(...)` menyusun token hasil stemming sesuai urutan aslinya.

Mengapa cache penting?

- Satu kata seperti `pengiriman` bisa muncul puluhan kali.
- Tanpa cache, Sastrawi akan menghitung ulang setiap kemunculan.
- Dengan cache, kata yang sama cukup dihitung **satu kali**.
- Pada dataset ini ada sekitar 151.000 token setelah stopword removal, tetapi
  hanya 12.436 kata unik. Cache menghemat sekitar 139.000 pemanggilan stemming.

### 4.6 Menggabungkan Semua Tahap

```python
def preprocess(teks):
    # Menjalankan seluruh tahap preprocessing secara berurutan
    if not isinstance(teks, str):
        return []

    teks = case_folding(teks)
    teks = cleaning(teks)
    tokens = tokenisasi(teks)
    tokens = hapus_stopword(tokens)
    tokens = stemming(tokens)

    return tokens
```

Penjelasan:

- `isinstance(teks, str)` memastikan input berupa teks. Jika ada nilai kosong
  atau bukan string, fungsi mengembalikan list kosong agar tidak error.
- Urutan pemanggilan mengikuti alur preprocessing:
  1. case folding,
  2. cleaning,
  3. tokenisasi,
  4. stopword removal,
  5. stemming.
- Hasil akhir berupa **list token bersih**.

### 4.7 Menerapkan ke Seluruh Dataset

```python
semua_tokens = []

for i, teks in enumerate(df["review"], start=1):
    semua_tokens.append(preprocess(teks))

    if i % 1000 == 0:
        print(f"{i} ulasan selesai diproses...")

df["tokens_bersih"] = pd.Series(semua_tokens, index=df.index, dtype="object")
df["teks_bersih"] = df["tokens_bersih"].apply(lambda token: " ".join(token))
```

Penjelasan:

- `enumerate(..., start=1)` memberi nomor urut mulai dari 1.
- Setiap ulasan diproses dengan `preprocess`, lalu hasilnya dimasukkan ke list
  `semua_tokens`.
- `if i % 1000 == 0` menampilkan progres setiap 1000 ulasan agar kita tahu
  proses masih berjalan.
- `pd.Series(..., dtype="object")` menyimpan list token per baris. `dtype="object"`
  penting agar pandas tidak mencoba memecah list menjadi kolom baru.
- `" ".join(token)` menggabungkan kembali token menjadi satu string, sehingga
  mudah dibaca di tabel.

Proses ini adalah bagian paling lama karena stemming. Pada dataset ini hasil
akhirnya: **11.606 ulasan selesai** dan **12.436 kata unik** masuk ke cache.

---

## 5. Tabel 10 Contoh Sebelum dan Sesudah

```python
contoh_tabel = df.sample(10, random_state=42)[["sentimen", "review", "teks_bersih"]].copy()
contoh_tabel["sentimen"] = contoh_tabel["sentimen"].map({0: "Negatif", 1: "Positif"})
contoh_tabel.insert(0, "No", range(1, 11))
contoh_tabel = contoh_tabel.rename(columns={
    "sentimen": "Sentimen",
    "review": "Sebelum",
    "teks_bersih": "Sesudah"
})
contoh_tabel
```

Penjelasan:

- `sample(10, random_state=42)` mengambil 10 baris acak.
- `random_state=42` membuat hasil acak selalu sama setiap kali dijalankan
  (reproducible).
- `[["sentimen", "review", "teks_bersih"]]` memilih kolom yang ditampilkan.
- `.copy()` membuat salinan agar DataFrame asli tidak ikut berubah.
- `map({0: "Negatif", 1: "Positif"})` mengubah kode angka menjadi nama label.
- `insert(0, "No", range(1, 11))` menambahkan kolom nomor urut di paling kiri.
- `rename(...)` mengganti nama kolom agar lebih mudah dibaca.
- `display.max_colwidth` diatur 100 agar teks panjang tidak terlalu cepat
  terpotong pada tampilan tabel.

---

## 6. Statistik Sebelum vs Sesudah

```python
kata_sebelum = df["review"].str.lower().str.split()
rata_kata_sebelum = kata_sebelum.apply(len).mean()
vocab_sebelum = set(kata for tokens in kata_sebelum for kata in tokens)
```

Penjelasan:

- `str.lower()` menyeragamkan huruf sebelum menghitung kosa kata.
- `str.split()` memecah teks asli menjadi kata berdasarkan spasi.
- `.apply(len).mean()` menghitung panjang setiap baris lalu mengambil rata-rata.
- `set(...)` mengumpulkan semua kata dari seluruh baris. Karena set tidak
  menyimpan duplikat, hasilnya adalah **kosa kata unik**.
- Pada teks asli, tanda baca masih menempel pada kata, sehingga variasi kata
  terlihat lebih banyak.

```python
rata_kata_sesudah = df["tokens_bersih"].apply(len).mean()
vocab_sesudah = set(kata for tokens in df["tokens_bersih"] for kata in tokens)
```

Penjelasan:

- `tokens_bersih` sudah berupa list token hasil preprocessing.
- `.apply(len)` menghitung jumlah token per ulasan.
- Kosa kata unik dihitung dari seluruh token bersih.

```python
statistik = pd.DataFrame({
    "Tahap": ["Sebelum preprocessing", "Sesudah preprocessing"],
    "Rata-rata jumlah kata": [round(rata_kata_sebelum, 2), round(rata_kata_sesudah, 2)],
    "Jumlah kosa kata unik": [len(vocab_sebelum), len(vocab_sesudah)]
})
statistik
```

Penjelasan:

- Tabel ini membandingkan dua kondisi: teks asli dan teks bersih.
- `round(..., 2)` membulatkan rata-rata menjadi dua angka di belakang koma.
- `len(vocab_...)` menghitung banyaknya kata unik.

```python
selisih = len(vocab_sebelum) - len(vocab_sesudah)
persen_turun = selisih / len(vocab_sebelum) * 100
print(f"Kosa kata unik berkurang {selisih} kata ({persen_turun:.2f}%).")
```

Penjelasan:

- `selisih` adalah jumlah kata unik yang hilang.
- `persen_turun` menghitung persentase penurunan.
- `f"..."` adalah f-string untuk menyisipkan nilai ke dalam teks.
- `:.2f` menampilkan dua angka di belakang koma.

### Hasil pada Dataset Ini

| Tahap | Rata-rata jumlah kata | Jumlah kosa kata unik |
|---|---:|---:|
| Sebelum preprocessing | 15,76 | 23.371 |
| Sesudah preprocessing | 13,06 | 10.161 |

Kosa kata unik berkurang **13.210 kata (56,52%)**. Rata-rata jumlah kata hanya
turun sedikit karena cleaning dan stopword removal membuang sebagian kata, tetapi
stemming menggabungkan banyak bentuk kata yang berbeda menjadi satu kata dasar.

---

## 7. Troubleshooting Google Colab

### `ModuleNotFoundError: No module named 'datasets'` atau `'Sastrawi'`

Penyebab: sel **Persiapan Library** belum dijalankan atau instalasi gagal.

Solusi:

1. Jalankan sel `%pip install -q datasets Sastrawi`.
2. Tunggu sampai instalasi selesai.
3. Jika Colab meminta restart, pilih **Runtime -> Restart session**.
4. Jalankan **Run all** lagi.

### `NameError: name 'tokenisasi' is not defined`

Penyebab: sel demo dijalankan sebelum sel definisi fungsi.

Solusi: jalankan sel definisi `def tokenisasi(...)` terlebih dahulu, atau gunakan
**Runtime -> Run all** agar urutannya benar.

### `You must restart the runtime in order to use newly installed versions`

Penyebab: Colab memperbarui library yang sudah terpasang.

Solusi: **Runtime -> Restart session**, lalu **Run all** lagi. Data yang sudah
dihitung akan hilang, jadi proses stemming akan diulang.

### Dataset gagal diunduh / error koneksi

Penyebab: internet terputus atau HuggingFace membatasi permintaan.

Solusi:

1. Pastikan internet stabil.
2. Jalankan ulang sel pemuatan dataset.
3. Tunggu beberapa saat lalu coba lagi.

### Proses stemming lama atau terputus

Penyebab: Sastrawi harus memeriksa aturan imbuhan untuk ribuan kata unik.

Solusi:

1. Tunggu sampai muncul `Selesai! Total: 11606 ulasan`.
2. Jangan menutup tab saat proses berjalan.
3. Jika sesi terputus, sambungkan ulang dan **Run all** lagi.

### Grafik tidak muncul

Penyebab: sel grafik belum dijalankan atau output sel terhapus.

Solusi: jalankan ulang sel grafik. Di Colab, grafik akan tampil otomatis setelah
`plt.show()`.

---

## 8. Tiga Pertanyaan Latihan

1. Mengapa case folding sebaiknya dilakukan sebelum cleaning dan tokenisasi?
   Apa yang bisa salah jika urutannya dibalik?
2. Stopword removal Sastrawi ikut membuang kata negasi seperti "tidak" dan
   "bukan". Jelaskan dampaknya untuk analisis sentimen, lalu usulkan solusi.
3. Lihat tabel statistik. Mengapa jumlah kosa kata unik turun jauh lebih besar
   dibandingkan rata-rata jumlah kata per ulasan? Kaitkan jawabanmu dengan
   peran stemming dan stopword removal.
