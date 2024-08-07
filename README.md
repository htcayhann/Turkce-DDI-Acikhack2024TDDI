# Turkce DDI Acikhack2024TDDI
 Teknofest Anlam Avcıları takımı olarak hazırladığımız Türkçe haber verileri üzerinden yalan haber tespiti, katagory sınıflandırma gibi DDİ işlemlerinin yapıldığı yarışma projesidir.

# Türkçe Haber Verilerinde Yalan Haber ve Kategori Tespiti Projesi

Projemize Türkçe haber veri seti eksikliğini fark ederek başladık ve bu amaçla kategori ve doğruluk etiketleri olan bir veri setioluşturduk. Haber verilerimizi DDİ (Doğal Dil İşleme) yöntemleri ile temizleyip tokenize ettik ve duygu analizini çıkardık. Ardından eğitim için uygun modelleri denedik. Naive Bayes gibi klasik yöntemlerde 0 kategorisinin tahmininde başarıya ulaşamadık. Sadece Bert modeli de özellikle 0 kategorisi tahmininde başarılı olamadı. Gerekli literatür taramalarından sonra BERT-CNN hibrit modeli ile başarılı sonuç elde edebildik ve doğruluk modelinde doğruluk oranımız %81,25 ; kategori modelinde doğruluk oranımız %84,7 olarak bulunmuştur.

Haber Doğruluk Modeli Sonuçları:




![image](https://github.com/user-attachments/assets/8dd98b7d-367a-4d28-a585-2091252faf07)


Kategori Modeli Sonuçları:




![image](https://github.com/user-attachments/assets/e6c271a6-ffb6-4e54-a97f-866696ee34e1)





## Özellikler

- Türkçe veri seti oluşturulmuştur.
- Türkçe dil bilgisi yapısına uygun tokenizasyon için Stanza kullanılmıştır.
- Tokenize edilmiş veri üzerinde sözlük kullanılarak duygu analizi yapılmıştır.
- Model eğitimleri için BERT-CNN tabanlı hibrit model oluşturulmuş ve yüksek doğruluk oranları elde edilmiştir.

## Gereksinimler

Bu projeyi çalıştırmak için gerekli bağımlılıklar `requirements.txt` dosyasında listelenmiştir. Proje çalıştırılmak istendiğinde:

```bash
pip install -r requirements.txt

komutu ile ortama yüklenebilir.




 ## Çalıştırma Adımları
1.	Veri Toplama ve Hazırlık:
o data.py: News API aracılığıyla veri çekilmesi.
o	veriHazirlama.py: Verilerin gerekli formatta birleştirilmesi.
2.	Veri Temizleme:
o	veri_temizleme.py: Temizleme işlemi ve nihai veri oluşturma.
3.	Tokenizasyon ve Lemmatizasyon:
o	stanza_tokken.py: Türkçe dil bilgisi yapısına göre tokenize ve lemmatize edilmesi.
4.	Duygu Analizi:
o	VADER.py: SWNetTR kütüphanesi ve VADER ile duygu analizi yapılması.
5.	Model Eğitimi:
o	Daha fazla 0 etiketli haberlere ihtiyaç duyulduğu için teyit.org gibi sitelerden 0 etiketli veriler toplanmış ve model_egitim_veri.csv oluşturulmuştur.
o	Model eğitimi için category_model.py ve bert_cnn_haber_model.py dosyasını kullanın.


## KAYNAKÇA

-	Yapay Zekâ Tabanlı Doğal Dil İşleme Yaklaşımını Kullanarak
İnternet Ortamında Yayınlanmış Sahte Haberlerin Tespiti
https://dergipark.org.tr/en/download/article-file/1817671

-	Türkçe Haberlerin Tür Tespiti İçin Konu
Modelleme Yöntemlerinin Karşılaştırılması
https://www.researchgate.net/profile/Zekeriya-Gueven-2/publication/337526948_Comparison_of_Topic_Modeling_Methods_for_Type_Detection_of_Turkish_News/links/5de2c18e299bf10bc334f04c/Comparison-of-Topic-Modeling-Methods-for-Type-Detection-of-Turkish-News.pdf


-	Türkçe Haber Başlıklarından Konu Tespiti
https://www.researchgate.net/profile/Cengiz-Hark/publication/366153944_Turkce_Haber_Basliklarindan_Konu_Tespiti_Topic_Detection_from_Turkish_News_Texts/links/63934467e42faa7e75aced35/Tuerkce-Haber-Basliklarindan-Konu-Tespiti-Topic-Detection-from-Turkish-News-Texts.pdf

-	Twitter'da Makine Öğrenmesi Yöntemleriyle Sahte Haber Tespiti
https://dergipark.org.tr/en/download/article-file/3015090

-	BERT-CNN: Improving BERT for Requirements Classification using CNN
https://www.sciencedirect.com/science/article/pii/S187705092300234X?ref=pdf_download&fr=RR-2&rr=8af977b178d23632

-	KUISAIL at SemEval-2020 Task 12: BERT-CNN for Offensive Speech Identification in Social Media
https://aclanthology.org/2020.semeval-1.271.pdf
