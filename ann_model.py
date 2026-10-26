import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

print("--- 1. ADIM: Veri Yükleniyor ve Hazırlanıyor ---")
fashion_mnist = tf.keras.datasets.fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Normalizasyon (Pikselleri 0.0 - 1.0 aralığına çekme)
X_train = X_train / 255.0
X_test = X_test / 255.0

print("\n--- 2. ADIM: ANN Model Mimarisi Oluşturuluyor ---")
model_ann = models.Sequential([
    # 28x28 pikseli 784 elemanlı tek boyutlu vektöre dönüştürür
    layers.Flatten(input_shape=(28, 28)),
    
    # 128 nöronlu ilk gizli katman
    layers.Dense(128, activation='relu'),
    
    # 64 nöronlu ikinci gizli katman
    layers.Dense(64, activation='relu'),
    
    # 10 sınıf için olasılık üreten çıktı katmanı
    layers.Dense(10, activation='softmax')
])

# Modelin özetini ekrana basar
model_ann.summary()

print("\n--- 3. ADIM: Model Derleniyor (Compile) ---")
model_ann.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n--- 4. ADIM: Model Eğitiliyor (Training) ---")
# Epochs=10: Veri setinin tamamını 10 kez baştan sona inceleyecek
history_ann = model_ann.fit(
    X_train, y_train,
    epochs=10,
    validation_data=(X_test, y_test)
)

print("\n--- 5. ADIM: Başarım Grafikleri Çizdiriliyor ---")
plt.figure(figsize=(12, 4))

# Doğruluk (Accuracy) Grafiği
plt.subplot(1, 2, 1)
plt.plot(history_ann.history['accuracy'], label='Eğitim Başarımı (Train Acc)')
plt.plot(history_ann.history['val_accuracy'], label='Test Başarımı (Val Acc)')
plt.title('ANN Model Doğruluğu (Accuracy)')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

# Kayıp (Loss) Grafiği
plt.subplot(1, 2, 2)
plt.plot(history_ann.history['loss'], label='Eğitim Kaybı (Train Loss)')
plt.plot(history_ann.history['val_loss'], label='Test Kaybı (Val Loss)')
plt.title('ANN Model Kaybı (Loss)')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()

plt.show()

# Eğitilen modeli ileride CNN ile karşılaştırmak üzere kaydediyoruz
model_ann.save('ann_fashion_model.h5')
print("\nANN Modeli başarıyla eğitildi ve 'ann_fashion_model.h5' olarak kaydedildi!")