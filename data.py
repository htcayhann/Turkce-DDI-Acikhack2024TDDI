import requests
import pandas as pd
from collections import defaultdict

# API anahtarınızı buraya girin
api_key = 'API-KEY'

# API URL'sini oluşturun
url = 'https://newsapi.org/v2/everything'

# Kategori anahtar kelimelerini tanımla
categories = {
    'politics': [
        'politik', 'seçim', 'hükümet', 'Erdoğan', 'muhalefet', 'mülteci', 'göçmen', 'Filistin',
        'Netanyahu', 'Trump', 'Amerika', 'Biden', 'İsrail', 'Gazze', 'TBMM', 'parti', 'demokrasi',
        'anayasa', 'yasa', 'kanun', 'protesto', 'miting', 'parlamento', 'başkanlık', 'meclis', 'koalisyon', 'diplomatik'
    ],
    'sports': [
        'spor', 'futbol', 'basketbol', 'voleybol', 'Fenerbahçe', 'Galatasaray', 'Beşiktaş', 'olimpiyat',
        'TFF', 'milli takım', 'maç', 'şampiyona', 'turnuva', 'hakem', 'transfer', 'antrenör', 'sporcu',
        'atletizm', 'yarış', 'lig', 'derbi', 'şampiyon', 'sakatlık'
    ],
    'technology': [
        'teknoloji', 'yapay zeka', 'bilgisayar', 'Bitcoin', 'Yazılım', 'Microsoft', 'Google', 'Meta',
        'Twitter', 'robotik', 'inovasyon', 'uygulama', 'akıllı telefon', 'dijital', 'bulut bilişim',
        'siber güvenlik', 'veri bilimi', 'blokzincir', 'kriptografi', 'algoritma', 'otomasyon', 'uzay'
    ],
    'entertainment': [
        'eğlence', 'dizi', 'oyuncu', 'film', 'Instagram', 'müzik', 'pop', 'konser', 'magazin', 'fenomen',
        'şarkıcı', 'aktör', 'aktris', 'televizyon', 'sanatçı', 'sosyal medya', 'video', 'tiyatro', 'sinema',
        'albüm', 'kültür'
    ],
    'economics': [
        'economics', 'Altın', 'Dolar', 'Banka', 'Faiz', 'Euro', 'Küresel Piyasa', 'Merkez Bankası', 'Kredi',
        'Finans', 'borsa', 'yatırım', 'hisse senedi', 'kripto para', 'ekonomi', 'enflasyon', 'döviz', 'ticaret',
        'işsizlik', 'gelir', 'vergi', 'bütçe', 'para politikası'
    ],

}

def get_news(api_key, query, language='tr', page_size=100, page=1):
    params = {
        'q': query,
        'language': language,
        'pageSize': page_size,
        'page': page,  # Sayfa numarası
        'apiKey': api_key
    }
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"API isteği başarısız oldu. Durum kodu: {response.status_code}")
        print(f"Hata mesajı: {response.json()}")
        return None
    data = response.json()
    return data

def categorize_article(title, description, threshold=1):
    category_scores = defaultdict(int)
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword.lower() in title.lower():
                category_scores[category] += 1
            if keyword.lower() in description.lower():
                category_scores[category] += 1

    # Eşik değer belirleme
    if category_scores:
        max_category = max(category_scores, key=category_scores.get)
        if category_scores[max_category] >= threshold:
            return max_category
    return 'other'

# Parametrelerle veri çekme
query = 'Dünya'  # Anahtar kelimeyi düzenledik,5 Fraklı query ile veri cekip daha sonra birlestirdik, bu bir ornegi
language = 'tr'
page_size = 50
total_articles = []  # Tüm makaleleri saklamak için

# Belirli sayfa sayısı kadar döngü oluşturma 
for page in range(1, 2):  
    news_data = get_news(api_key, query, language, page_size, page)

    if news_data and news_data.get('status') == 'ok':
        # Gelen haberleri işleme
        articles = news_data.get('articles', [])
        for article in articles:
            category = categorize_article(article['title'], article['description'])
            total_articles.append({
                'title': article['title'],
                'description': article['description'],
                'publishedAt': article['publishedAt'],
                'source': article['source']['name'],
                'url': article['url'],
                'category': category  # Kategori bilgisini ekle
            })
    else:
        print("Veri alınamadı veya sonlandırılıyor.")
        break

# Veriyi bir DataFrame'e dönüştür
df = pd.DataFrame(total_articles)

# Veriyi CSV dosyasına kaydet
df.to_csv('world.csv', index=False, encoding='utf-8-sig')
print('Veriler .csv dosyasına kaydedildi.')
