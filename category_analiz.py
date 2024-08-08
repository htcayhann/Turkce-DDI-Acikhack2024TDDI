import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# CSV dosyası
texts_path = r'desc_title_vader_results.csv'
df = pd.read_csv(texts_path, encoding='utf-8-sig')


if 'category' in df.columns and 'sentiment_label' in df.columns and 'sentiment_score' in df.columns:

    def plot_category_sentiment_distribution(df, title, output_file):
        sentiment_labels = ['negatif', 'pozitif', 'nötr']
        categories = df['category'].unique()


        sentiment_distribution = {cat: df[df['category'] == cat]['sentiment_label'].value_counts() for cat in categories}

        sentiment_distribution = {cat: dist for cat, dist in sentiment_distribution.items() if dist.sum() > 0}

        filtered_categories = list(sentiment_distribution.keys())
        index = np.arange(len(filtered_categories))

        # Grafik verileri
        bar_width = 0.2
        colors = ['red','purple', 'blue']

        plt.figure(figsize=(12, 8))

        for i, sentiment in enumerate(sentiment_labels):
            plt.bar(index + i * bar_width,
                    [sentiment_distribution[cat].get(sentiment, 0) for cat in filtered_categories],  # NaN veya 0 değerleri atlama
                    bar_width, label=sentiment.capitalize(), color=colors[i])

        plt.xlabel('Kategori')
        plt.ylabel('Haber Sayısı')
        plt.title(title)
        plt.xticks(index + bar_width, filtered_categories, rotation=45)  # Kategorileri x ekseninde göster
        plt.legend(title="Duygu Durumu")

        plt.tight_layout()
        plt.savefig(output_file, format='png')
        plt.close()

    plot_category_sentiment_distribution(df, 'Kategorilere Göre Duygu Durumu Dağılımı',
                                         'category_sentiment_distribution.png')
    print("Kategorilere göre duygu durumu dağılımı grafiği kaydedildi.")

else:
    print("'category', 'sentiment_label' ve 'sentiment_score' sütunları bulunamadı. Analiz yapılamıyor.")
