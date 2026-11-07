| Kondisi citra                            |   Jumlah citra | Perilaku model yang teramati                                                                                                                                 | Contoh image_id    |
|:-----------------------------------------|---------------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------|:-------------------|
| (a) Tumpukan penuh, media tidak terlihat |              0 | Model mensegmentasi hampir seluruh tumpukan feses lama sebagai foreground karena ketiadaan fitur pembeda antara ekskreta segar vs latar kotoran.             | -                  |
| (b) Tumpukan sebagian                    |             30 | Model mampu mendeteksi gugus tumpukan kotoran dan memisahkannya dari lantai kandang yang terbuka, namun terjadi over-segmentation pada batas tumpukan basah. | IMG_3028, IMG_3030 |
| (c) Ekskreta segar terpisah              |              0 | Model bekerja optimal sesuai asumsi pelatihan; ekskreta segar terisolasi berhasil disegmentasi dengan batas tepi yang presisi.                               | -                  |

Overlay dapat diakses di link:

https://drive.google.com/drive/folders/1gyUBvPQ6--uKlmcuw5HY_reQxBkoBp-P?usp=sharing