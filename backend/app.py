from flask import Flask, request, jsonify, send_file, render_template_string
from flask_cors import CORS
import cv2
import numpy as np
import os
from PIL import Image
import base64
import io
from datetime import datetime
import torch
import torchvision.transforms as transforms
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import requests
from pathlib import Path

app = Flask(__name__)
CORS(app)

# Konfigürasyon
UPLOAD_FOLDER = '../uploads'
STATIC_FOLDER = '../static'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Klasörlerin varlığını kontrol et
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# SAM2 Model Cache
sam2_predictor = None

def download_sam2_model():
    """SAM2 model dosyasını indir"""
    model_dir = Path("../models")
    model_dir.mkdir(exist_ok=True)
    
    checkpoint_path = model_dir / "sam2_hiera_tiny.pt"
    
    if checkpoint_path.exists():
        print("DEBUG: SAM2 model dosyası zaten var.")
        return str(checkpoint_path)
    
    print("DEBUG: SAM2 model dosyası indiriliyor... (Bu işlem biraz zaman alabilir)")
    
    # SAM2 Tiny model (daha küçük ve hızlı)
    model_url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
    
    try:
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        downloaded = 0
        
        with open(checkpoint_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        print(f"DEBUG: İndirme %{percent:.1f} tamamlandı", end='\r')
        
        print(f"\nDEBUG: Model dosyası başarıyla indirildi: {checkpoint_path}")
        return str(checkpoint_path)
        
    except Exception as e:
        print(f"HATA: Model dosyası indirilemedi: {e}")
        return None

def load_sam2_model():
    """SAM2 modelini yükle (sadece bir kez)"""
    global sam2_predictor
    if sam2_predictor is None:
        try:
            print("DEBUG: SAM2 modeli yükleniyor...")
            
            # Model dosyasını indir/bul
            checkpoint_path = download_sam2_model()
            if checkpoint_path is None:
                raise ValueError("Model dosyası bulunamadı")
            
            # SAM2 konfigürasyonu (Tiny model için)
            model_cfg = "sam2_hiera_t.yaml"
            
            # Model oluştur
            sam2_model = build_sam2(model_cfg, checkpoint_path, device="cpu")
            sam2_predictor = SAM2ImagePredictor(sam2_model)
            
            print("DEBUG: SAM2 modeli başarıyla yüklendi!")
            return True
            
        except Exception as e:
            print(f"UYARI: SAM2 modeli yüklenemedi: {e}")
            print("DEBUG: GrabCut algoritması kullanılacak...")
            return False
    return True

def create_sam2_mask(image_path, x, y, width, height):
    """
    SAM2 modeli ile akıllı segmentasyon yaparak binary mask oluşturur
    """
    try:
        # SAM2 modelini yükle
        if not load_sam2_model():
            raise ValueError("SAM2 modeli yüklenemedi")
        
        # Resmi yükle
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Resim yüklenemedi")
        
        # BGR'den RGB'ye çevir
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width = image_rgb.shape[:2]
        
        print(f"DEBUG: SAM2 - Resim boyutları: {img_width}x{img_height}")
        print(f"DEBUG: SAM2 - Gelen koordinatlar: x={x}, y={y}, width={width}, height={height}")
        
        # Koordinatları sınırla
        x = max(0, min(x, img_width - 1))
        y = max(0, min(y, img_height - 1))
        width = max(1, min(width, img_width - x))
        height = max(1, min(height, img_height - y))
        
        print(f"DEBUG: SAM2 - Düzeltilmiş koordinatlar: x={x}, y={y}, width={width}, height={height}")
        
        # SAM2 için resmi ayarla
        sam2_predictor.set_image(image_rgb)
        
        # Bounding box koordinatlarını SAM2 formatına çevir [x1, y1, x2, y2]
        input_box = np.array([x, y, x + width, y + height])
        
        # SAM2 ile mask üret
        masks, scores, logits = sam2_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],  # Box formatında
            multimask_output=False,
        )
        
        # En iyi mask'i seç (ilk mask, çünkü multimask_output=False)
        best_mask = masks[0]
        
        # Boolean mask'i binary mask'e çevir (0-255)
        binary_mask = (best_mask * 255).astype(np.uint8)
        
        print(f"DEBUG: SAM2 - Mask oluşturuldu, beyaz piksel sayısı: {np.sum(binary_mask == 255)}")
        print(f"DEBUG: SAM2 - Toplam piksel sayısı: {binary_mask.size}")
        print(f"DEBUG: SAM2 - Confidence score: {scores[0]:.3f}")
        
        return binary_mask
        
    except Exception as e:
        print(f"HATA: SAM2 segmentasyon başarısız: {e}")
        raise e

def create_binary_mask(image_path, x, y, width, height):
    """
    GrabCut algoritması ile akıllı segmentasyon yaparak binary mask oluşturur
    x, y: Sol üst köşe koordinatları (bounding box)
    width, height: Seçilen alanın boyutları
    """
    # Resmi yükle
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Resim yüklenemedi")
    
    # Resim boyutlarını al
    img_height, img_width = image.shape[:2]
    
    print(f"DEBUG: Resim boyutları: {img_width}x{img_height}")
    print(f"DEBUG: Gelen koordinatlar: x={x}, y={y}, width={width}, height={height}")
    
    # Koordinatları sınırla ve doğrula
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    width = max(1, min(width, img_width - x))
    height = max(1, min(height, img_height - y))
    
    print(f"DEBUG: Düzeltilmiş koordinatlar: x={x}, y={y}, width={width}, height={height}")
    
    try:
        # GrabCut algoritması ile akıllı segmentasyon
        # Mask tanımla (0: kesin arka plan, 1: kesin ön plan, 2: muhtemelen arka plan, 3: muhtemelen ön plan)
        mask = np.zeros((img_height, img_width), np.uint8)
        
        # Bounding box tanımla
        rect = (x, y, width, height)
        
        # GrabCut için gerekli matrisler
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        # GrabCut algoritmasını çalıştır
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        
        # Mask'i binary formata çevir (0 ve 2: arka plan, 1 ve 3: ön plan)
        binary_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
        
        print(f"DEBUG: GrabCut tamamlandı, beyaz piksel sayısı: {np.sum(binary_mask == 255)}")
        print(f"DEBUG: Toplam piksel sayısı: {binary_mask.size}")
        
        return binary_mask
        
    except Exception as e:
        print(f"HATA: GrabCut başarısız oldu: {e}")
        print("DEBUG: Basit dikdörtgen mask'e geri dönülüyor...")
        
        # Hata durumunda basit dikdörtgen mask yap
        simple_mask = np.zeros((img_height, img_width), dtype=np.uint8)
        simple_mask[y:y+height, x:x+width] = 255
        
        print(f"DEBUG: Basit mask oluşturuldu, beyaz piksel sayısı: {np.sum(simple_mask == 255)}")
        
        return simple_mask

@app.route('/')
def index():
    """Ana sayfa"""
    with open('../frontend/index.html', 'r', encoding='utf-8') as f:
        return f.read()

@app.route('/upload', methods=['POST'])
def upload_image():
    """Resim upload endpoint"""
    if 'image' not in request.files:
        return jsonify({'error': 'Resim bulunamadı'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Dosya seçilmedi'}), 400
    
    if file:
        # Dosya adını güvenli hale getir
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"image_{timestamp}_{file.filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Dosyayı kaydet
        file.save(filepath)
        
        # Resim boyutlarını al
        with Image.open(filepath) as img:
            width, height = img.size
        
        return jsonify({
            'success': True,
            'filename': filename,
            'width': width,
            'height': height,
            'url': f'/uploads/{filename}'
        })

@app.route('/generate_mask', methods=['POST'])
def generate_mask():
    """Binary mask oluşturma endpoint"""
    data = request.json
    
    required_fields = ['filename', 'x', 'y', 'width', 'height']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} eksik'}), 400
    
    try:
        # Dosya yolunu oluştur
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Resim bulunamadı'}), 404
        
        # Binary mask oluştur (GrabCut ile)
        mask = create_binary_mask(
            image_path,
            int(data['x']),
            int(data['y']),
            int(data['width']),
            int(data['height'])
        )
        
        # Mask'i kaydet
        mask_filename = f"grabcut_mask_{data['filename']}"
        mask_path = os.path.join(STATIC_FOLDER, mask_filename)
        cv2.imwrite(mask_path, mask)
        
        return jsonify({
            'success': True,
            'mask_filename': mask_filename,
            'mask_url': f'/static/{mask_filename}',
            'method': 'grabcut'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_sam2_mask', methods=['POST'])
def generate_sam2_mask():
    """SAM2 ile binary mask oluşturma endpoint"""
    data = request.json
    
    required_fields = ['filename', 'x', 'y', 'width', 'height']
    for field in required_fields:
        if field not in data:
            return jsonify({'error': f'{field} eksik'}), 400
    
    try:
        # Dosya yolunu oluştur
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], data['filename'])
        
        if not os.path.exists(image_path):
            return jsonify({'error': 'Resim bulunamadı'}), 404
        
        # SAM2 ile binary mask oluştur
        mask = create_sam2_mask(
            image_path,
            int(data['x']),
            int(data['y']),
            int(data['width']),
            int(data['height'])
        )
        
        # Mask'i kaydet
        mask_filename = f"sam2_mask_{data['filename']}"
        mask_path = os.path.join(STATIC_FOLDER, mask_filename)
        cv2.imwrite(mask_path, mask)
        
        return jsonify({
            'success': True,
            'mask_filename': mask_filename,
            'mask_url': f'/static/{mask_filename}',
            'method': 'sam2'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Upload edilen dosyaları serve et"""
    return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename))

@app.route('/static/<filename>')
def static_file(filename):
    """Static dosyaları serve et"""
    return send_file(os.path.join(STATIC_FOLDER, filename))

@app.route('/download_mask/<filename>')
def download_mask(filename):
    """Mask dosyasını indirme endpoint"""
    file_path = os.path.join(STATIC_FOLDER, filename)
    if os.path.exists(file_path):
        return send_file(file_path, as_attachment=True)
    return jsonify({'error': 'Dosya bulunamadı'}), 404

if __name__ == '__main__':
    print("Binary Mask Server başlatılıyor...")
    print("Tarayıcınızda http://localhost:5001 adresini açın")
    app.run(debug=True, host='0.0.0.0', port=5001)