# RAG Sistemi

Bu proje, bilgisayarinizdaki PDF dosyalarini okuyup, icine sorular sorabileceginiz yerel bir yapay zeka sistemidir. Sistem sadece PDF icerigini kullanir, disaridan bilgi eklemez.

## Sistem Mimarisi

![RAG Architecture](https://github.com/It-Does-Not-Actually-Matter/papagan_rag/raw/main/rag.jpg)

Sistem 5 temel asamadan olusur:
1. **Document Loading**: PDF dosyalari yuklenir
2. **Splitting**: Belgeler kucuk parcalara bolunur
3. **Storage**: Parcalar vektorstore'da saklanir
4. **Retrieval**: Sorguya en uygun parcalar bulunur
5. **Output**: LLM ile cevap uretilir

## Ne Yapar?

- `data/` klasorundeki ilk 5 PDF dosyasini otomatik yukler
- Metinleri 800 karakterlik parcalara ayirir (120 karakter bindirme ile)
- BAAI/bge-m3 modeli ile embeddings olusturur
- Chroma vektorstore kullanarak benzerligi hesaplar
- Sorularinizi PDF icerigine gore cevaplar
- PDF'te yoksa su cevabi verir: `Baglamda cevap bulunamadi.`

## Gereksinimler

### Ollama Kurulu Olmali

Bu proje LLM olarak Ollama kullanir. Calistirmadan once:

```bash
ollama pull llama3:8b
```

Ollama arka planda calisir durumda olmali.

### Python Kutuphaneleri

Sanal ortam olusturun ve aktif edin:

```bash
python -m venv env
source env/bin/activate  # Linux/Mac
# veya
env\Scripts\activate  # Windows
```

Kutuphaneleri yukleyin:

```bash
pip install -r requirements.txt
```

## Kurulum

1. Projeyi klonlayin:
```bash
git clone https://github.com/It-Does-Not-Actually-Matter/papagan_rag.git
cd papagan_rag
```

2. `data/` klasoru olusturun ve PDF dosyalarinizi icine atin:
```bash
mkdir -p data
# PDF dosyalarinizi data/ klasorune kopyalayin
```

3. Sanal ortam olusturun ve bagimlilikari yukleyin:
```bash
python -m venv env
source env/bin/activate  # Linux/Mac veya env\Scripts\activate (Windows)
pip install -r requirements.txt
```

4. Ollama'nin calistigini dogrulayin:
```bash
ollama list
```

## Kullanim

Projeyi calistirin:

```bash
python main.py
```

Terminal acildiginda sorular sorabilirsiniz:

```
---RAG---
Kullanici: Yazilim tasarimi nedir?
Cevap: ...

Kullanici: Bu belgede ne anlatiliyor?
Cevap: ...

Kullanici: exit
```

Cikmak icin `exit` yazin.

## Teknik Detaylar

- **Embedding Model**: BAAI/bge-m3
- **LLM**: Llama3 8B (Ollama)
- **Vector Store**: Chroma
- **Chunk Size**: 800 karakter
- **Chunk Overlap**: 120 karakter
- **Retrieval**: Top-6 similarity search
- **Temperature**: 0.3 (daha deterministik cevaplar)

## Ozellikler

- Yerel calisir, internet gerektirmez
- Sadece verdiginiz PDF'lerden cevap verir
- Turkce cevaplar uretir (ASCII karakterler ile)
- Chroma DB persist ile tekrar yukleme gerektirmez
- Hallusinasyon onleme mekanizmasi

## Dosya Yapisi

```
papagan_rag/
├── main.py              # Ana uygulama kodu
├── requirements.txt     # Python bagimliliklar
├── README.md           # Dokumantasyon
├── data/               # PDF dosyalari (sizin olusturacaginiz)
├── chroma_db/          # Vektorstore (otomatik olusur)
└── .env                # Ortam degiskenleri (opsiyonel)
```

## Lisans

MIT License