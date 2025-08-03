# Binary Mask Generator 🎭

Resim üzerinde seçilen alanlar için binary mask oluşturan localhost web uygulaması.

## Özellikler ✨

- 📁 Resim upload (JPG, PNG, GIF)
- 🖱️ Mouse ile alan seçimi (box selection)
- 🎭 Binary mask oluşturma (seçilen alan beyaz, dış kısım siyah)
- ⬇️ Mask indirme
- 🖥️ Basit ve kullanışlı web arayüzü

## Kurulum 🚀

### 1. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 2. Sunucuyu Başlat

```bash
cd backend
python app.py
```

### 3. Tarayıcıda Aç

http://localhost:5000 adresini tarayıcınızda açın.

## Kullanım 📖

1. **Resim Yükle**: "Resim Seç" butonuna tıklayarak bir resim yükleyin
2. **Alan Seç**: Yüklenen resim üzerinde mouse ile sürükleyerek dikdörtgen alan seçin
3. **Mask Oluştur**: "Mask Oluştur" butonuna tıklayın
4. **İndir**: Oluşturulan binary mask'i indirin

## Proje Yapısı 📁

```
Binary Mask/
├── backend/
│   └── app.py              # Flask backend API
├── frontend/
│   └── index.html          # Web arayüzü
├── uploads/                # Yüklenen resimler
├── static/                 # Oluşturulan mask'ler
├── requirements.txt        # Python bağımlılıkları
└── README.md              # Bu dosya
```

## API Endpoints 🔌

- `GET /` - Ana sayfa
- `POST /upload` - Resim upload
- `POST /generate_mask` - Binary mask oluşturma
- `GET /uploads/<filename>` - Upload edilen resimleri serve et
- `GET /static/<filename>` - Oluşturulan mask'leri serve et
- `GET /download_mask/<filename>` - Mask indirme

## Teknik Detaylar ⚙️

- **Backend**: Flask + OpenCV + NumPy
- **Frontend**: HTML5 Canvas + JavaScript
- **Resim İşleme**: OpenCV ile binary mask oluşturma
- **File Upload**: 16MB'a kadar dosya desteği

## Binary Mask Algoritması 🧠

1. Yüklenen resmin boyutları alınır
2. Kullanıcının seçtiği koordinatlar (x, y, width, height) işlenir
3. Resim boyutunda siyah (0) bir mask matrisi oluşturulur
4. Seçilen alan koordinatları beyaz (255) ile doldurulur
5. Sonuç PNG formatında kaydedilir

## Özelleştirme 🛠️

- Maksimum dosya boyutu: `app.py` içinde `MAX_CONTENT_LENGTH`
- Canvas max genişlik: `index.html` içinde `maxWidth`
- Port numarası: `app.py` içinde `port=5000`

## Geliştirme Notları 📝

- SAM2 entegrasyonu için backend'e model yükleme fonksiyonu eklenebilir
- Çoklu seçim desteği için frontend genişletilebilir
- Farklı mask formatları (JPG, BMP) için export seçenekleri eklenebilir
- Batch processing için API endpoint'leri eklenebilir
