import cv2
import numpy as np
from PIL import Image
import os

# --- INISIALISASI ---
path = 'dataset' 
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# --- FUNGSI MENGAMBIL DATA DAN LABEL ---
def getImagesAndLabels(path):
    # Mengambil semua folder ID (1, 2, dst) dari folder dataset
    # Perbaikan pada baris ini agar tidak ada syntax error
    id_folders = [os.path.join(path, f) for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]
    
    faceSamples = []
    ids = []
    
    for folderPath in id_folders:
        # Mengambil angka ID dari nama folder (misal folder '1' jadi ID 1)
        try:
            id_user = int(os.path.basename(folderPath)) 
        except ValueError:
            print(f"Melewati folder '{os.path.basename(folderPath)}' karena bukan angka.")
            continue
        
        # Iterasi melalui semua gambar dalam folder ID tersebut
        for filename in os.listdir(folderPath):
            full_path = os.path.join(folderPath, filename)
            
            if full_path.endswith(('.jpg', '.png', '.jpeg')):
                try:
                    PIL_img = Image.open(full_path).convert('L') # Ubah ke grayscale
                    img_numpy = np.array(PIL_img, 'uint8')

                    faces = detector.detectMultiScale(img_numpy)

                    for (x,y,w,h) in faces:
                        faceSamples.append(img_numpy[y:y+h,x:x+w])
                        ids.append(id_user)
                except Exception as e:
                    print(f"Melewati file rusak: {full_path}. Error: {e}")

    return faceSamples, ids

# --- PROSES PELATIHAN (TRAINING) ---
print("\n [INFO] Melatih wajah. Tunggu sebentar ...")

# Pastikan folder dataset ada
if not os.path.exists(path):
    print(f"\n [ERROR] Folder '{path}' tidak ditemukan!")
else:
    faces, ids = getImagesAndLabels(path)

    # Pastikan ada data untuk dilatih
    if len(faces) > 0:
        recognizer.train(faces, np.array(ids))

        # Pastikan folder 'trainer' ada sebelum menyimpan
        if not os.path.exists('trainer'):
            os.makedirs('trainer')
            
        # Simpan model
        recognizer.write('trainer/trainer.yml') 

        print(f"\n [INFO] {len(np.unique(ids))} wajah dilatih. File 'trainer/trainer.yml' berhasil dibuat.")
    else:
        print("\n [ERROR] Tidak ada sampel wajah ditemukan di folder dataset.")