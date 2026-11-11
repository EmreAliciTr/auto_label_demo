# Üçüncü taraf bildirimleri

| Bileşen | Lisans beyanı | Kaynak |
|---|---|---|
| Grounding DINO / Grounding DINO Tiny | Apache-2.0 | https://github.com/IDEA-Research/GroundingDINO ve https://huggingface.co/IDEA-Research/grounding-dino-tiny |
| PP-OCRv6 medium | Apache-2.0 | https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det_safetensors ve https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec_safetensors |
| Transformers | Apache-2.0 | https://github.com/huggingface/transformers |
| PyTorch / Torchvision | BSD ve birlikte sunulan bildirimler | https://github.com/pytorch/pytorch ve https://github.com/pytorch/vision |
| NAFNet / OpenCV ONNX | MIT / BasicSR Apache-2.0 | https://github.com/megvii-research/NAFNet ve https://huggingface.co/opencv/deblurring_nafnet |
| Concertormer | MIT / BasicSR Apache-2.0 | https://github.com/setsunil/Concertormer |

Apache-2.0 metni `lisanslar/APACHE-2.0.txt` içinde; NAFNet ve Concertormer'ın
MIT/BasicSR Apache bildirimleri `lisanslar/GORUNTU_IYILESTIRME-LISANS.txt` içindedir.
Sabit model sürümleri `ortak/model_surumleri.json` dosyasındadır.

Concertormer mimarisi `goruntu_iyilestirme/concertormer_mimarisi/` içinde bulunur. Kaynak sürümü
`94269c556277805051644343e3a2cc76c4cd7aaf`; bağımsız yükleme için importlar
uyarlanmıştır. Model sahiplerinin telif ve lisans bildirimleri geçerlidir.

Python bağımlılıklarının kendi LICENSE/NOTICE dosyaları ayrıca geçerlidir.
Paket veya ağırlıklar yeniden dağıtıldığında bunlarla gelen bildirimler korunmalıdır.
