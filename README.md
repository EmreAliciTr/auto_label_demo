# Etiketleme ve görüntü iyileştirme demoları

İki ayrı amaç için arayüz gerektirmeyen Python backend kodları:

| Yapmak istediğiniz işlem | Klasör ve kullanım açıklaması | Çalıştırma dosyası |
|---|---|---|
| Nesne kutusu bulmak veya yazı okumak | [etiketleme/](etiketleme/README.md) — Grounding DINO Tiny, PP-OCRv6 medium | [etiketleme_calistir.py](etiketleme/etiketleme_calistir.py) |
| Bulanıklık ve JPEG bozulmasını azaltmak | [goruntu_iyilestirme/](goruntu_iyilestirme/README.md) — NAFNet-REDS, Concertormer-REDS | [goruntu_iyilestirme_calistir.py](goruntu_iyilestirme/goruntu_iyilestirme_calistir.py) |

**Etiketleme çıktısı:** Kutu, metin ve skor içeren JSON. Bu sonuçlar ham model
tahminidir; onaylı etiket veya YOLO eğitim dosyası oluşturulmaz.
**Görüntü iyileştirme çıktısı:** Girdiyle aynı piksel boyutunda iyileştirilmiş PNG.

## Klasör düzeni

```text
etiketleme/             Nesne bulma ve OCR kodları, çalıştırıcı, kullanım
goruntu_iyilestirme/    Netleştirme kodları, çalıştırıcı, üç örnek görsel, kullanım
ortak/                 Ortak model indirme/doğrulama, servis ve HTTP kodları
lisanslar/             Üçüncü taraf lisans metinleri
tests/                 Testler ve test bağımlılıkları
yerel_api_baslat.py    İki bölüm için isteğe bağlı ortak localhost API
```

`etiketleme_motorlari.py` nesne bulma/OCR uygulamasıdır;
`iyilestirme_motorlari.py` NAFNet/Concertormer uygulamasıdır.
İlgili bölümdeki `*_calistir.py` dosyası komut satırı girişidir.

## Ortak kurulum

**Bütün komutları bu README'nin bulunduğu demo kök klasöründe çalıştırın.**
Python 3.12 ve etkinleştirilmiş ayrı bir sanal ortam kullanılması önerilir.

```sh
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements.txt
```

Ardından kullanacağınız bölümün README dosyasındaki model hazırlama ve çalıştırma
adımlarını izleyin. Görüntü iyileştirmenin ek bağımlılıkları kendi klasöründedir.
Bir bölümde `--engine all`, yalnız o bölümdeki iki motoru kapsar.

İlk paket ve model indirmesi internet gerektirir. Ağırlıklar kaynak paketine dahil
değildir. Hazır modellerle çıkarım çevrimdışı yapılır. Ortak varsayılan model
klasörü `models/`; başka bir klasör için `--models` kullanılır. Sabit sürüm ve
dosya hash'leri [ortak/model_surumleri.json](ortak/model_surumleri.json) içindedir.

## İsteğe bağlı ortak HTTP API

Doğrudan komut satırı kullanımı için bu sunucu gerekli değildir.

```sh
python yerel_api_baslat.py --port 8090
```

Sunucu `http://127.0.0.1:8090` adresinde çalışır. Kapatmak için Ctrl+C.
Başka bir terminalde, yine demo kök klasöründe:

```sh
python ortak/api_istek_ornegi.py --engine ppocr-v6-medium --image inputs/etiketleme_ornegi.png
python ortak/api_istek_ornegi.py --engine nafnet-reds --image goruntu_iyilestirme/ornek_gorseller/01_orijinal.png --output outputs/nafnet_iyilestirilmis.png
```

`GET /health` dört motor adını listeler; bu, ağırlıkların yüklenmiş olduğunu
göstermez. `POST /predict` etiketleme bölümünü, `POST /restore` görüntü
iyileştirme bölümünü çalıştırır. Yalnız seçilen model ilk istekte yüklenir,
sonraki isteklerde bellekte kullanılır.

`/predict` gövdesi: `engine`, `image_base64`; isteğe bağlı `prompt`, `targets`,
`threshold`, `text_threshold`, `nms`. `/restore` gövdesi: `engine`, `image_base64`,
`blend`; sonuç PNG'si yanıttaki `image_base64` alanındadır.
Sunucu uygulaması [ortak/yerel_api_sunucusu.py](ortak/yerel_api_sunucusu.py),
Python servis katmanı [ortak/motor_servisi.py](ortak/motor_servisi.py) içindedir.

Genel girdi sınırları: tek kare JPEG/PNG, 16 MiB, 20 milyon piksel, en fazla
8192 piksel kenar. HTTP gövde sınırı 24 MiB'dır. Görüntü iyileştirmede bölüm
README'sindeki daha düşük piksel sınırları da uygulanır.

## Testler ve lisanslar

```sh
python -m pip install -r tests/requirements.txt
python -B -m unittest discover -s tests -v
```

İki bölümün testlerini birlikte çalıştırmak için görüntü iyileştirmenin ek
bağımlılıklarını da kurun.
[Lisanslar ve kaynaklar](LISANSLAR_VE_KAYNAKLAR.md).
