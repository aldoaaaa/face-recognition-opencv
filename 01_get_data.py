import cv2
import os

# !!! 1. PENGATURAN ID (Ubah angka ini sesuai ID orang yang ingin diambil gambarnya) !!!
id_wajah = 3

# Inisialisasi kamera
# !!! 2. BARIS INI YANG HARUS DIUBAH JIKA KAMERA TIDAK MUNCUL !!!
# Menggunakan indeks 0 (default) dan memaksa backend DSHOW untuk Windows
cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# Cek apakah kamera berhasil dibuka (Tambahkan ini di bawah inisialisasi kamera)
if not cam.isOpened():
    print("-------------------------------------------------------------------")
    print("!!! ERROR FATAL: Gagal membuka kamera. Coba ganti angka 0 menjadi 1 atau 2. !!!")
    print("-------------------------------------------------------------------")
    # Hentikan program jika gagal
    exit()
cam.set(3, 640) # Lebar
cam.set(4, 480) # Tinggi

# Muat detektor wajah
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

print(f"\n [INFO] Mulai mengambil data wajah untuk ID: {id_wajah}. Lihat ke kamera!")

count = 0 

# Pastikan folder dataset/ID tersedia
folder_path = os.path.join('dataset', str(id_wajah))
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# --- LOOP PENGAMBILAN GAMBAR ---
while(True):
    ret, img = cam.read()
    # img = cv2.flip(img, -1) # Buka komentar ini jika kamera terbalik
    
    # Cek apakah frame berhasil dibaca
    if not ret:
        print("\n [ERROR] Gagal membaca frame dari kamera. Cek koneksi kamera.")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1

        # Simpan gambar wajah yang terpotong ke folder dataset/ID
        cv2.imwrite(f"{folder_path}/User.{id_wajah}.{count}.jpg", gray[y:y+h,x:x+w])

        # Tampilkan teks bantuan di jendela kamera
        cv2.putText(img, f"ID: {id_wajah} | Sampel: {count}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        
    cv2.imshow('Image Taker - Tekan Q untuk Keluar', img)

    k = cv2.waitKey(100) & 0xff 
    # Keluar jika tombol 'q' ditekan ATAU sudah mengambil 50 gambar
    if k == ord('q') or count >= 50:
        break

# --- PENUTUPAN ---
print("\n [INFO] Pengambilan data selesai.")
cam.release()
cv2.destroyAllWindows()