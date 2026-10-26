import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf 
import matplotlib.pyplot as plt
import numpy as np

# TensorFlow'un sunucularından Fashion-MNIST veri setini çekiyoruz.
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# X_train: Eğitim resimleri, y_train: Bu resimlerin ne olduğunu gösteren etiketler (0'dan 9'a kadar sayılar)
print(f"Eğitimdeki Resim Sayısı ve Boyutu: {X_train.shape}")
# Çıktı (60000, 28, 28) olacak -> 60.000 adet 28x28 piksel resim var demektir.

print(f"Testteki Resim Sayısı ve Boyutu: {X_test.shape}")
# Çıktı (10000, 28, 28) olacak -> 10.000 adet test resmi.

#İlk resmin (0. indeksteki resim) ilk 5 satır ve 5 sütunundaki piksel değerlerine bakalım:
print("İlk resmin sol üst köşesindeki pikseller (0-255 arası değerler):")
print(X_train[0][:5, :5])

# Bilgisayarda etiketler sayı olarak tutulur (0, 1, 2... 9). 
# İnsanların anlaması için bu sayıların hangi kıyafete denk geldiğini listeliyoruz:
class_names = ['Tişört/Üst', 'Pantolon', 'Kazak', 'Elbise', 'Palto',
               'Sandalet', 'Gömlek', 'Spor Ayakkabı', 'Çanta', 'Bilekte Bot']


print(f"\nİlk resmin etiket numarası: {y_train[0]}")
print(f"Bu numaranın karşılığı olan kıyafet: {class_names[y_train[0]]}")

print("\n--- 4. ADIM: Normalizasyon (Pikselleri 0-1 Arasına Çekme) ---")
# 0-255 arasındaki piksel değerlerini 255'e bölerek 0.0 - 1.0 aralığına getiriyoruz.
# Neden? Yapay sinir ağları küçük sayılarla çok daha hızlı ve doğru öğrenir!
X_train_norm = X_train / 255.0
X_test_norm = X_test / 255.0

# Veri setindeki ilk 10 resmi yan yana bir pencerede görelim:
plt.figure(figsize=(12, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])  # X eksenindeki çizgileri kaldırır
    plt.yticks([])  # Y eksenindeki çizgileri kaldırır
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary) # Siyah-beyaz gösterim
    plt.xlabel(class_names[y_train[i]]) # Resmin altına kıyafet adını yaz
plt.suptitle("Fashion-MNIST Veri Setinden İlk 10 Örnek Görsel", fontsize=14)
plt.show() # Pencereyi açar 