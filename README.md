# Face Recognition dengan OpenCV

Sistem pengenalan wajah sederhana menggunakan Haar Cascade untuk
deteksi wajah dan LBPH untuk pengenalan.

## Cara kerja
1. `01_get_data.py` mengambil foto wajah lewat kamera
2. `train.py` melatih model LBPH dari dataset
3. `face_detector.py` mengenali wajah secara real-time

## Cara menjalankan
pip install -r requirements.txt
python 01_get_data.py
python train.py
python face_detector.py

## Catatan
Dataset tidak disertakan karena berisi data wajah pribadi.
Kumpulkan dataset sendiri dengan izin orang yang bersangkutan.

## Teknologi
Python, OpenCV, NumPy, Pillow