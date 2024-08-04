import csv
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from collections import defaultdict



def load_sentiment_lexicon(file_path):
    lexicon = {}
    try:
        df = pd.read_csv(file_path, sep=';', decimal=",", encoding="utf-8-sig")

        for _, row in df.iterrows():
            word = row.get('WORD')
            polarity_str = str(row.get('POLARITY', '0')).replace(',', '.')
            try:
                polarity = float(polarity_str)
                if word:
                    lexicon[word] = polarity
            except ValueError as e:
                print(f"Polarity değeri hatalı: {e} | Kelime: {word}, POLARITY: {polarity_str}")
    except FileNotFoundError:
        print(f"Dosya bulunamadı: {file_path}")
    except Exception as e:
        print(f"Bir hata oluştu: {e}")

    if lexicon:
        print(f"Sözlükteki ilk 10 kelime ve skoru: {list(lexicon.items())[:10]}")
    else:
        print("Sözlük boş veya veriler okunamadı.")

    return lexicon


def update_vader_lexicon(vader_analyzer, custom_lexicon):
    for word, score in custom_lexicon.items():
        vader_analyzer.lexicon[word] = score


def create_custom_analyzer(custom_lexicon_path):
    analyzer = SentimentIntensityAnalyzer()
    custom_lexicon = load_sentiment_lexicon(custom_lexicon_path)
    update_vader_lexicon(analyzer, custom_lexicon)
    return analyzer


def analyze_sentiment(text, analyzer):
    sentiment = analyzer.polarity_scores(text)
    compound_score = sentiment['compound']

    if compound_score >= 0.05:
        sentiment_label = 'pozitif'
    elif compound_score == 0.00:
        sentiment_label = 'nötr'
    else:
        sentiment_label = 'negatif'

    return compound_score, sentiment_label


custom_lexicon_path = r'SWNetTR++.csv'  # Özelleştirilmiş sözlüğünüzün dosya yolu
analyzer = create_custom_analyzer(custom_lexicon_path)

texts_path = r'model_egitim_veri.csv'  # Analiz edilecek metinlerin dosya yolu
columns_to_analyze = ['temizlenmis_title_lemmas', 'temizlenmis_description_lemmas']
category_column = 'category'  # Kategori sütununun ismi

# Sonuçları kaydedeceğimiz dosya
output_file_path = r'sentiment_analysis_results1.csv'

# Kategori bazında ortalama duygu skorlarını hesaplamak için
category_scores = defaultdict(lambda: {'total_score': 0, 'count': 0})

with open(texts_path, mode='r', encoding='utf-8') as file, open(output_file_path, mode='w', newline='',
                                                                encoding='utf-8') as output_file:
    reader = csv.DictReader(file)
    writer = csv.writer(output_file)

    writer.writerow(['Text', 'Category', 'Score', 'Label'])

    for row in reader:
        category = row.get(category_column, 'Unknown')

        for column in columns_to_analyze:
            text = row.get(column, '').strip()
            score, label = analyze_sentiment(text, analyzer)
            writer.writerow([text, category, score, label])  # Sonuçları yaz

            # Kategori bazında skoru güncelle
            category_scores[category]['total_score'] += score
            category_scores[category]['count'] += 1

with open(output_file_path, mode='a', newline='', encoding='utf-8') as output_file:
    writer = csv.writer(output_file)
    writer.writerow([])
    writer.writerow(['Kategori', 'Ortalama Duygu Skoru', 'Genel Duygu Durumu'])

    for category, scores in category_scores.items():
        if scores['count'] > 0:
            avg_score = scores['total_score'] / scores['count']
            if avg_score >= 0.05:
                overall_sentiment = 'pozitif'
            elif avg_score == 0.00:
                overall_sentiment = 'nötr'
            else:
                overall_sentiment = 'negatif'
        else:
            avg_score = 0
            overall_sentiment = 'belirsiz'

        writer.writerow([category, avg_score, overall_sentiment])
