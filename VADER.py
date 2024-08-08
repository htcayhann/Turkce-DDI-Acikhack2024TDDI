import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


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

custom_lexicon_path = r'SWNetTR++.csv'  
analyzer = create_custom_analyzer(custom_lexicon_path)

texts_path = r'model_egitim_veri.csv'  # Analiz edilecek metinler

df = pd.read_csv(texts_path, encoding='utf-8')

df['combined_text'] = df['temizlenmis_title_lemmas'].fillna('') + ' ' + df['temizlenmis_description_lemmas'].fillna('')

df['sentiment_score'] = df['combined_text'].apply(lambda x: analyze_sentiment(x, analyzer)[0])
df['sentiment_label'] = df['combined_text'].apply(lambda x: analyze_sentiment(x, analyzer)[1])

output_path = r'desc_title_vader_results.csv'  
df.to_csv(output_path, index=False, encoding='utf-8-sig')
print(f"Sonuçlar {output_path} dosyasına kaydedildi.")
