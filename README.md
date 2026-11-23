# Preprocessing Teks Ulasan E-commerce Berbahasa Indonesia

- **Nama:** ARIL AL IQBAL
- **NIM:** 24146034

## Deskripsi Project

Project ini melakukan **preprocessing teks ulasan produk e-commerce berbahasa
Indonesia** menggunakan dataset HuggingFace
[`dipawidia/ecommerce-product-reviews-sentiment`](https://huggingface.co/datasets/dipawidia/ecommerce-product-reviews-sentiment).

Dataset yang dipakai adalah split `train` dengan 11.606 ulasan dan tiga kolom:
`review`, `translate`, dan `sentimen`. Label `0` berarti negatif dan `1` berarti
positif.

Tahapan yang dikerjakan:

1. Memuat dataset dengan library `datasets`, lalu mengubahnya menjadi DataFrame pandas.
2. Menampilkan jumlah data, nama kolom, serta distribusi label (angka, persentase,
   dan grafik batang).
3. Preprocessing teks dalam fungsi terpisah:
   - case folding,
   - cleaning (URL, angka, tanda baca, karakter khusus),
   - tokenisasi,
   - stopword removal dengan Sastrawi,
   - stemming dengan Sastrawi dan cache per kata unik.
4. Menampilkan 10 contoh teks sebelum dan sesudah preprocessing dalam satu tabel.
5. Menghitung rata-rata jumlah kata dan jumlah kosa kata unik sebelum vs sesudah
   preprocessing.

## File Project

| File                                   | Keterangan                                     |
| -------------------------------------- | ---------------------------------------------- |
| `preprocessing_ulasan_ecommerce.ipynb` | Notebook utama tugas                           |
| `PENJELASAN_KODE.md`                   | Penjelasan per bagian kode dan troubleshooting |
| `README.md`                            | Deskripsi project dan cara menjalankan         |

## Cara Menjalankan

### Google Colab

1. Upload `preprocessing_ulasan_ecommerce.ipynb` ke Google Colab.
2. Pilih **Runtime -> Restart session** agar sesi bersih.
3. Jalankan sel **Persiapan Library**. Sel ini berisi:

   ```python
   %pip install -q datasets Sastrawi
   ```

4. Pilih **Runtime -> Run all**.
5. Jika Colab menampilkan pesan **"You must restart the runtime"**, pilih
   **Runtime -> Restart session**, lalu **Run all** lagi.

### Komputer Lokal

```bash
pip install datasets pandas matplotlib Sastrawi jupyter
jupyter notebook preprocessing_ulasan_ecommerce.ipynb
```

Jalankan semua sel dari atas ke bawah. Jangan menjalankan sel demo sebelum sel
definisi fungsinya.

## Hasil Singkat

| Keterangan                    | Hasil                |
| ----------------------------- | -------------------- |
| Jumlah ulasan                 | 11.606               |
| Negatif (0)                   | 5.803 (50,00%)       |
| Positif (1)                   | 5.803 (50,00%)       |
| Rata-rata jumlah kata sebelum | 15,76                |
| Rata-rata jumlah kata sesudah | 13,06                |
| Kosa kata unik sebelum        | 23.371               |
| Kosa kata unik sesudah        | 10.161               |
| Penurunan kosa kata unik      | 13.210 kata (56,52%) |

## Catatan

- Notebook membutuhkan koneksi internet untuk mengunduh dataset dari HuggingFace.
- Proses stemming pada seluruh data bisa memakan beberapa menit. Tunggu sampai
  muncul tulisan `Selesai! Total: 11606 ulasan`.
- Daftar stopword Sastrawi juga memuat kata negasi seperti "tidak". Untuk analisis
  sentimen, kata negasi sebaiknya dipertahankan karena dapat membalik makna.
- Pada teks asli, tanda baca masih menempel pada kata, sehingga jumlah kosa kata
  "sebelum" terlihat lebih besar. Perbandingan ini tetap berguna untuk menunjukkan
  efek pembersihan.
