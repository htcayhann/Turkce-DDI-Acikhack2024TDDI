import pandas as pd
import re

# Türkçe stop words seti
stop_words = {'a', 'ab', 'abi', 'ad', 'ama', 'ancak', 'arada', 'az', 'bazen', 'belki', 'bir', 'biri', 'birkaç',
              'birçok', 'bu', 'bunu', 'bunun', 'da', 'de', 'demek', 'değil', 'diğer', 'daha', 'de', 'değil',
              'diğer', 'dolayı', 'çok', 'hem', 'her', 'hiç', 'ile', 'işte', 'kadar', 'ki', 'la', 'le', 'mu', 'mü', 'ne',
              'neden', 'niye', 'nü', 'o', 'öyle', 'şey', 'şimdi', 've', 'veya', 'ya', 'yani', 'zaten'}

# Temizleme fonksiyonu
def temizle(text):
    if pd.isna(text):
        return ""
    text = text.lower()  # Büyük harfleri küçük harfe çevir
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    text = re.sub(r'\d+', '', text)  # Sayıları kaldır
    text = re.sub(r'\s+', ' ', text).strip()  # Gereksiz boşlukları kaldır
    return text

# Stop words kaldırma fonksiyonu
def stop_words_kaldir(text):
    return ' '.join([word for word in text.split() if word not in stop_words])

# CSV dosyasını yükleme
dosya_adi = 'yeniDosya.csv'
df = pd.read_csv(dosya_adi)

# Eksik değerleri doldurma
df['title'] = df['title'].fillna('')
df['description'] = df['description'].fillna('')

# Temizleme ve stop words çıkarma uygulama
df['temizlenmis_title'] = df['title'].apply(lambda x: stop_words_kaldir(temizle(x)))
df['temizlenmis_description'] = df['description'].apply(lambda x: stop_words_kaldir(temizle(x)))

# Hem temizlenmis_title hem de temizlenmis_description boş olan satırları kaldırma
df = df[~((df['temizlenmis_title'] == '') & (df['temizlenmis_description'] == ''))]

# Temizlenmiş veri setini kaydetme
df.to_csv('temizlenmis_veri_seti.csv', index=False)

print("Veri temizleme işlemi tamamlandı ve temizlenmiş veri seti 'temizlenmis_veri_seti.csv' olarak kaydedildi.")
print(df.head())
