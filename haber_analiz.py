import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

texts_path = r'desc_title_vader_results.csv'
df = pd.read_csv(texts_path, encoding='utf-8-sig')

if 'Real/Fake' in df.columns and 'sentiment_label' in df.columns  and 'sentiment_score' in df.columns:

    def plot_fake_news_sentiment_distribution(df, title, output_file):
        sentiment_labels = ['negatif', 'pozitif', 'nötr']
        categories = [0, 1]  # 0: Yalan Haber, 1: Gerçek Haber

        sentiment_distribution = {
            0: df[df['Real/Fake'] == 0]['sentiment_label'].value_counts().reindex(sentiment_labels).fillna(0),
            1: df[df['Real/Fake'] == 1]['sentiment_label'].value_counts().reindex(sentiment_labels).fillna(0)
        }

        # Grafik verileri
        bar_width = 0.25
        index = np.arange(len(categories))
        colors = ['blue', 'yellow', 'green']

        plt.figure(figsize=(10, 6))


        for i, sentiment in enumerate(sentiment_labels):
            plt.bar(index + i * bar_width, [sentiment_distribution[cat][sentiment] for cat in categories],
                    bar_width, label=sentiment.capitalize(), color=colors[i])

        plt.xlabel('Haber Türü')
        plt.ylabel('Haber Sayısı')
        plt.title(title)
        plt.xticks(index + bar_width, ['Yalan Haber', 'Gerçek Haber'])
        plt.legend(title="Duygu Durumu")

        plt.tight_layout()
        plt.savefig(output_file, format='png')
        plt.close()


    plot_fake_news_sentiment_distribution(df, 'Yalan ve Gerçek Haberlerin Duygu Durumu Dağılımı',
                                          'fake_news_sentiment_distribution.png')
    print("Yalan ve gerçek haberlerin duygu durumu dağılımı grafiği kaydedildi.")

else:
    print("'Real/Fake', 'sentiment_label', 'sentiment_score' sütunu bulunamadı. Analiz yapılamıyor.")
