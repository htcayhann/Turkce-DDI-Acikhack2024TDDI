import pandas as pd
import stanza

# Türkçe dil modelini indirin ve yükleyin
stanza.download('tr')
nlp = stanza.Pipeline('tr')

# CSV dosyasını okuyun
csv_file = 'dogru_Kullanim.csv'
df = pd.read_csv(csv_file)

# İşlemek istediğiniz sütunları seçin
text_columns = ['temizlenmis_title', 'temizlenmis_description']  

# Her iki sütunda da işlemleri gerçekleştirin
for column in text_columns:
    df[column + '_tokens'] = df[column].apply(lambda x: [word.text for word in nlp(x).iter_words()])
    df[column + '_lemmas'] = df[column].apply(lambda x: [word.lemma for word in nlp(x).iter_words()])

# İşlenmiş verileri bir CSV dosyasına yazın
df.to_csv('nihai_veri.csv', index=False) #islem kolayligi elde etmek icin verimize lemmizasyon ve tokkenizasyon islemleri uygulandi ve onlar farkli bir dosyaya kaydedildi

print("İşlem tamamlandı ve sonuçlar dosyaya kaydedildi.")
