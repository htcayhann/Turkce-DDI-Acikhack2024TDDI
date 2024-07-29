import csv

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


def load_sentiment_lexicon(file_path):
    # Veriyi pandas ile yükleme
    try:
        df = pd.read_csv(file_path, sep=';', decimal=",", encoding="utf-8-sig")
        # Başlıkları ve ilk birkaç satırı kontrol et
        print(f"Başlıklar: {df.columns}")
        print(df.head())

        lexicon = {}
        for _, row in df.iterrows():
            word = row.get('WORD')
            tone_str = str(row.get('TONE', '0')).replace(',', '.')
            polarity_str = str(row.get('POLARITY', '0')).replace(',', '.')
            try:
                tone = float(tone_str)
                polarity = float(polarity_str)
                if word:
                    lexicon[word] = tone * polarity
            except ValueError as e:
                print(f"Veri dönüştürme hatası: {e} | Kelime: {word}, TONE: {tone_str}, POLARITY: {polarity_str}")
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {file_path}")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

    # Sözlükteki ilk 10 kelime ve skoru yazdır
    if lexicon:
        print(f"Sözlükteki ilk 10 kelime ve skoru: {list(lexicon.items())[:10]}")
    else:
        print("Sözlük boş veya veriler okunamadı.")

    return lexicon


def update_vader_lexicon(vader_analyzer, custom_lexicon):
    for word, score in custom_lexicon.items():
        vader_analyzer.lexicon[word] = score
    print(
        f"VADER sözlüğüne eklenen ilk 10 kelime ve skoru: {list(vader_analyzer.lexicon.items())[:10]}")  # İlk 10 kelime ve skoru yazdır


def create_custom_analyzer(custom_lexicon_path):
    analyzer = SentimentIntensityAnalyzer()
    custom_lexicon = load_sentiment_lexicon(custom_lexicon_path)
    update_vader_lexicon(analyzer, custom_lexicon)
    return analyzer


def analyze_sentiment(text, analyzer):
    text = text.lower()
    sentiment = analyzer.polarity_scores(text)
    print(f"Metin: {text} | Duygu Skoru: {sentiment['compound']}")
    return sentiment['compound']


# Özelleştirilmiş duygu sözlüğünü yükle
custom_lexicon_path = r'SWNetTR++.csv'  # Özelleştirilmiş sözlüğünüzün dosya yolu
analyzer = create_custom_analyzer(custom_lexicon_path)

# Analiz edilecek metinleri dosyadan yükleme
texts_path = r'nihai_veri.csv'  # Analiz edilecek metinlerin dosya yolu
columns_to_analyze = ['temizlenmis_title_lemmas', 'temizlenmis_description_lemmas',
                      'category']  # Analiz edilecek sütunların isimleri

with open(texts_path, mode='r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        for column in columns_to_analyze:
            text = row.get(column, '').strip()  # Boş string olarak varsayalım
            score = analyze_sentiment(text, analyzer)
            print(f"Sütun: {column}")
            print(f"Metin: {text}")
            print(f"Duygu Skoru: {score}")
            print("\n")

# Test için basit metinler
test_texts = [
    "Bu harika bir gün!",
    "Bu kötü bir deneyim.",
    "Bu oldukça nötr bir metin."
]

for text in test_texts:
    score = analyze_sentiment(text, analyzer)
    print(f"Test Metin: {text}")
    print(f"Duygu Skoru: {score}")
    print("\n")
