from PIL import Image
import os
import cv2
import numpy as np

# Mendapatkan daftar file dalam folder
folder_path = "E:\\dataset deep learning\\Stadium kanker payudara\\Dataset\\new\\grey\\2gg\\"
folder_path1 = "E:\\dataset deep learning\\Stadium kanker payudara\\Dataset\\new\\grey\\2g\\"
file_list = os.listdir(folder_path)

# Loop melalui setiap file dalam folder
for file_name in file_list:
    # Memeriksa apakah file adalah gambar dengan ekstensi yang didukung (misalnya .jpg, .jpeg, .png)
    if file_name.endswith((".jpg", ".jpeg", ".png")):
        # Membuka gambar RGB
        image_rgb = Image.open(os.path.join(folder_path, file_name))

        # Mengubah gambar ke grayscale
        image_gray = image_rgb.convert("L")

        # Mengubah gambar ke dalam bentuk array numpy
        np_image = np.array(image_gray)

        # Mengubah tingkat keabuan menggunakan tresholding
        threshold_value = 110  # Nilai ambang batas (threshold) yang digunakan
        _, image_thresholded = cv2.threshold(np_image, threshold_value, 255, cv2.THRESH_BINARY)

        # Mengubah nilai ambang batas (threshold) menjadi warna abu-abu
        image_thresholded_gray = np.where(image_thresholded == 255, threshold_value, image_thresholded)

        # Membuat objek gambar PIL dari array numpy thresholded_gray
        image_thresholded_gray_pil = Image.fromarray(image_thresholded_gray.astype(np.uint8))

        # Menyimpan gambar thresholded_gray dengan nama yang sama
        file_name_thresholded_gray = file_name.replace(".", "_thresholded_gray.")
        image_thresholded_gray_pil.save(os.path.join(folder_path1, file_name_thresholded_gray))
