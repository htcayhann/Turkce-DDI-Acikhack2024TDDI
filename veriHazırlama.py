#BU KODLAR VERİ SUTUN TEMIZLIGI VE CEKILEN VERILERIN BIRLESTIRILMESI ICIN GEREKLI OLAN KOD BLOGLARIDIR

import pandas as pd

# CSV dosyasını yükle ve gereksiz sütunları sil
def temizle_csv_dosyasi(giris_dosyasi, cikis_dosyasi):
    df = pd.read_csv(giris_dosyasi)
    df = df.drop(columns=['title', 'description', 'publishedAt'])
    df.to_csv(cikis_dosyasi, index=False)

# Birden fazla CSV dosyasını birleştir
def birlestir_csv_dosyalari(dosya_listesi, cikis_dosyasi):
    df_list = [pd.read_csv(dosya) for dosya in dosya_listesi]
    df_birlesmis = pd.concat(df_list)
    df_birlesmis.to_csv(cikis_dosyasi, index=False)

# Bir CSV dosyasındaki yinelenen satırları kaldır
def yinelenenleri_sil(giris_dosyasi, cikis_dosyasi):
    df = pd.read_csv(giris_dosyasi)
    df = df.drop_duplicates()
    df.to_csv(cikis_dosyasi, index=False)

# Bir CSV dosyasından belirtilen satırı sil
def satir_sil(giris_dosyasi, cikis_dosyasi, silinecek_satir):
    df = pd.read_csv(giris_dosyasi)
    df = df.drop(index=silinecek_satir)
    df.to_csv(cikis_dosyasi, index=False)

# İlk belirli sayıda satırı tut ve diğerlerini sil
def ilk_satirlari_tut(giris_dosyasi, cikis_dosyasi, satir_sayisi):
    df = pd.read_csv(giris_dosyasi)
    df = df.iloc[:satir_sayisi]
    df.to_csv(cikis_dosyasi, index=False)

# Örnek kullanım
# temizle_csv_dosyasi('nihai_veri.csv', 'nihai_Kullanim.csv')
# birlestir_csv_dosyalari(['guncel_dosya.csv', 'politics.csv', 'turkey.csv', 'world.csv'], 'Son.csv')
# yinelenenleri_sil('enSon.csv', 'enSonDrop.csv')
# satir_sil('yeniDosya.csv', 'yeni_dosya_adı.csv', 95)
# ilk_satirlari_tut('yeniDosya.csv', 'yeni_dosya_adı.csv', 467)
