import cv2
import numpy as np
import os 

# --- 1. INISIALISASI ---
# Muat model yang telah dilatih (hasil dari train.py)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);

font = cv2.FONT_HERSHEY_SIMPLEX

# --- 2. DAFTAR NAMA SESUAI ID (PENTING!) ---
# Index 0 = Unknown
# Index 1 = aldoa
# Index 2 = yuda
# Index 3 = hilal
names = ['Unknown', 'aldoa', 'yuda' , 'hilal']  # <<< UBAH BARIS INI!

# --- 3. INISIALISASI KAMERA ---
# ... kode lainnya
# Gunakan indeks kamera yang sebelumnya berhasil menyala (misalnya 1 atau 0)
# Jika masalah kamera muncul lagi, coba ganti angka 0 di bawah ini.
# Gunakan cv2.CAP_DSHOW jika trik sebelumnya berhasil.
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Menggunakan CAP_DSHOW karena lebih stabil di Windows
cam.set(3, 640) # Lebar frame
cam.set(4, 480) # Tinggi frame

# Atur ambang batas (threshold) kepercayaan (confidence)
# Angka yang lebih rendah berarti pengenalan lebih ketat/yakin.
confidence_threshold = 100 

# --- 4. LOOP DETEKSI REAL-TIME ---
while True:
    ret, img = cam.read()
    if not ret:
        print("Gagal membaca frame kamera.")
        break

    # img = cv2.flip(img, -1) # Buka komentar ini jika kamera terbalik
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (100, 100)
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        
        # Prediksi ID dan Confidence
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # --- 5. LOGIKA PENAMPILAN NAMA ---
        # Confidence yang lebih rendah berarti pengenalan lebih baik (0 adalah sempurna)
        if (confidence < confidence_threshold):
            # Jika confidence-nya bagus, gunakan nama yang sesuai dengan ID
            name = names[id]
            # Hitung persentase confidence (untuk debugging)
            confidence_display = "  {0:.2f}%".format(round(100 - confidence))
        else:
            # Jika confidence terlalu tinggi, anggap sebagai "Unknown"
            name = "Unknown"
            confidence_display = "  {0:.2f}%".format(round(100 - confidence))
        
        # Tampilkan nama dan confidence di atas kotak deteksi
        cv2.putText(img, str(name), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence_display), (x+5,y+h-5), font, 1, (255,255,0), 1)  

    # Tampilkan jendela hasil
    cv2.imshow('Deteksi Wajah Real-Time', img) 

    # Tekan 'q' untuk keluar
    k = cv2.waitKey(10) & 0xff 
    if k == ord('q'):
        break

# --- 6. PENUTUPAN ---
print("\n [INFO] Program deteksi selesai.")
cam.release()
cv2.destroyAllWindows()