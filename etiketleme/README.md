# Etiketleme — nesne bulma ve yazı okuma

- **Grounding DINO Tiny:** Yazılı nesne tarifinden kutular üretir.
- **PP-OCRv6 medium:** Türkçe ve çok dilli yazıları okuyup metin bölgelerini döndürür.

Giriş: JPEG/PNG. Çıkış: kutu, metin ve skor içeren JSON. Görsel netleştirme bu
bölümde yapılmaz. Çıktı, gözden geçirilmesi gereken ham model tahminidir.

## Dosyalar

| Dosya | Görevi |
|---|---|
| `etiketleme_calistir.py` | İki etiketleme motorunun komut satırı girişi |
| `etiketleme_motorlari.py` | DINO ve PP-OCR model yükleme/çıkarım kodu |
| `ornek_gorsel_olustur.py` | Deneme için yapay yazılı bir PNG oluşturur |

## Hazırlama

**Bu klasöre girmeden, demo kök klasöründe çalıştırın.** Önce
[ortak kurulumu](../README.md#ortak-kurulum) tamamlayın.

```sh
python etiketleme/etiketleme_calistir.py download --engine all --allow-network
python etiketleme/etiketleme_calistir.py check --engine all
python etiketleme/ornek_gorsel_olustur.py
```

`all` yalnız Grounding DINO ve PP-OCR modellerini kapsar. Tek motor için
`all` yerine `grounding-dino-tiny` veya `ppocr-v6-medium` yazın.

## Deneme

```sh
python etiketleme/etiketleme_calistir.py predict --engine ppocr-v6-medium --image inputs/etiketleme_ornegi.png
python etiketleme/etiketleme_calistir.py predict --engine ppocr-v6-medium --image inputs/etiketleme_ornegi.png --target "HELLO 123" --threshold 0.5
python etiketleme/etiketleme_calistir.py predict --engine grounding-dino-tiny --image inputs/etiketleme_ornegi.png --prompt "a product. text." --threshold 0.25
```

Kendi görselinizi `--image` ile seçin. DINO için `--prompt` zorunludur;
İngilizce nesne tarifleri önerilir. OCR'da `--target` verilmezse bütün yazılar
döner. Birden fazla hedef için parametre tekrarlanabilir; filtre Türkçe harf
kurallarıyla tam metin eşleşmesi yapar.

DINO bölgeleri `x1,y1,x2,y2,score`, OCR bölgeleri
`text,polygon,score,detector_score` içerir. Koordinatlar yönü düzeltilmiş görselin
piksellerine göredir. `--threshold` skor eşiği; `--text-threshold` ve `--nms`
DINO parametreleridir. JSON'u dosyaya kaydetmek için komut sonuna
`> etiketleme_sonucu.json` ekleyebilirsiniz.

## Python içinden

Demo kökünde çalışan bir Python betiğinde:

```python
from pathlib import Path
from ortak.motor_servisi import InferenceService

service = InferenceService(Path("models"))
result = service.predict(
    "ppocr-v6-medium",
    Path("inputs/etiketleme_ornegi.png").read_bytes(),
    threshold=0.5,
)
```

Servisi ayrı bir çıkarım sürecinde kullanın. HTTP kullanımının açıklaması
[ana README](../README.md#isteğe-bağlı-ortak-http-api) içindedir.
