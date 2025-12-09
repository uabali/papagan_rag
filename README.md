# RAG Sistemi

Bu proje, bilgisayarınızdaki PDF dosyalarını okuyup, içine sorular sorabileceğiniz yerel bir yapay zeka sistemidir. Sistem sadece PDF içeriğini kullanır, dışarıdan bilgi eklemez.

## Ne Yapar?

- `data/` klasöründeki ilk 5 PDF dosyasını otomatik yükler
- Metinleri parçalara ayırır
- Bu parçaları vektör haline getirir
- Sorularınızı PDF içeriğine göre cevaplar
- PDF'te yoksa şu cevabı verir: `Baglamda cevap bulunamadi.`

## Gereksinimler

### Ollama Kurulu Olmalı

Bu proje LLM olarak Ollama kullanır. Çalıştırmadan önce:

```bash
ollama pull llama3:8b