> not: bu repostory benim bitirme projemin geç oluşturulmuş public versiyonudur.

# MR GÖRÜNTÜLERİ İLE REKTAL KANSER TANISI VE MASKELENMESİ

> 3D MR Görüntülerinde Rektal Kanser Segmentasyonu için 3D U-Net Modeli ve Tkinter tabanlı Görsel Arayüz

Bu proje, 3 boyutlu Manyetik Rezonans (MR) görüntülerinde rektal kanserli dokuları otomatik olarak tespit etmek için geliştirilmiş bir **3D U-Net** derin öğrenme modelini ve bu modelin sonuçlarını analiz etmek için tasarlanmış bir görsel arayüzü içerir.

## 💡 Proje Hakkında

Bu çalışma, radyologlara tanı sürecinde yardımcı olmayı amaçlayan bir bilgisayar destekli tanı (CAD) sistemidir. Model, NIfTI (.nii) formatındaki 3D MR taramalarını girdi olarak alır ve potansiyel kanserli bölgeleri piksel bazlı olarak segmente eder (maskeler).

Geliştirilen masaüstü uygulama, kullanıcıların aynı anda üç farklı görünümü incelemesine olanak tanır:

1.  **Orijinal MR Görüntüsü:** İşlenmemiş ham 3D MR verisi.
2.  **Yapay Zeka Tahmini:** Modelin kanserli olarak tahmin ettiği bölgelerin boyanmış (segmente edilmiş) hali.
3.  **Temel Gerçek (Ground Truth):** Radyologlar tarafından manuel olarak işaretlenmiş, doğrulanmış kanser bölgesi.

## ✨ Temel Özellikler

  * **3D Segmentasyon:** 3D U-Net mimarisi sayesinde hacimsel (volumetrik) MR verilerinde yüksek doğruluklu tümör tespiti.
  * **Etkileşimli Arayüz:** Orijinal görüntü, model tahmini ve temel gerçeği yan yana karşılaştırma imkanı sunan basit bir masaüstü uygulaması.
  * **Akademik Temel:** Modelin doğruluğu, "rectal filling" (rektal dolum) gibi klinik öneme sahip verilerle zenginleştirilmiş bir eğitim setine dayanmaktadır.

## 🛠️ Kullanılan Teknolojiler

**Yapay Zeka Modeli**

  * **Framework:** **PyTorch**
  * **Medikal Görüntüleme:** **MONAI** (Medical Open Network for AI)
  * **Model Mimarisi:** **3D U-Net**
  * **Diğer Kütüphaneler:** NiBabel (NIfTI dosya işlemleri), NumPy

**Görüntüleme Uygulaması**

  * **Dil:** **Python**
  * **Arayüz (GUI):** **Tkinter**

## Alakalı bir porje

Projenin web üzerinde çalışan bir haline ![buradan](https://github.com/emirzaza/RectalCancerDiagnosisWEB) ulaşabilirsin. @emirzaza'ya teşekkürler.

## 🧠 Model ve Veri

### Veri Kaynağı ve Etik Sorumluluk

Bu model, hibrit bir veri seti kullanılarak eğitilmiştir:

1.  **Herkese Açık Veriler:** **Medical Decathlon Challenge** ([http://medicaldecathlon.com/](http://medicaldecathlon.com/)) platformundan elde edilen anonimleştirilmiş MR görüntüleri.
2.  **Özel Klinik Veriler:** İzmir Bakırçay Üniversitesi Hastanesi ile yürütülen araştırma kapsamında elde edilen, yüksek kalitede ve klinik olarak doğrulanmış özel veriler.

> **ÖNEMLİ NOT:** Etik kurallar ve hasta mahremiyeti (KVKK) gereğince, **üniversite hastanesinden elde edilen özel klinik veriler bu repoda paylaşılmamaktadır.**

### Eğitim Notu: Rektal Dolum (Rectal Filling)

Yapılan literatür taraması ve akademik araştırma sonucunda, "rectal filling" (rektal dolum) uygulanan hastalara ait MR verilerinin, modelin anatomik sınırları daha net ayırt etmesine ve dolayısıyla tümör tespit başarımını önemli ölçüde artırdığı gözlemlenmiştir. Eğitim verisi zenginleştirilirken bu faktör dikkate alınmıştır.
[Deep learning models for preoperative T-stage assessment in rectal cancer using MRI: exploring the impact of rectal filling](https://pmc.ncbi.nlm.nih.gov/articles/PMC10722089/)

## 💻 Kullanım

Uygulamayı başlatmak için terminal veya komut istemcisinden aşağıdaki komutu çalıştırın:

```bash
python MRIViewandSegapp.py
```

Arayüz açıldıktan sonra "MRI Yükle" butonu aracılığıyla `.nii` veya `.nii.gz` formatındaki 3D MR dosyanızı seçin. Görüntüyü inceleyebilir hale geliceksiniz, model tahmini ve temel gerçeği de ilgili butonlarla yükleyebilirsinz.

https://github.com/user-attachments/assets/49935037-8742-4120-b0a3-28f89bced29c

## 📧 İletişim

Proje hakkında bana ulaşmak isterseniz E-posta adresim: kaankazguc@hotmail.com
