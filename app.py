import os
import io
import logging
import sys
import base64
from datetime import datetime
from functools import wraps
from typing import Dict, Any, Tuple, Optional

import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import cv2

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    MODEL_PATH = os.getenv('MODEL_PATH', 'gastric_working_final.keras')
    IMG_SIZE = tuple(map(int, os.getenv('IMG_SIZE', '120,120').split(',')))
    THRESHOLD = float(os.getenv('PREDICTION_THRESHOLD', '0.4'))
    MAX_CONTENT_LENGTH = int(os.getenv('MAX_FILE_SIZE', 10 * 1024 * 1024))
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'tif', 'tiff', 'bmp'}
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    MIN_IMAGE_SIZE = 50
    HISTOLOGY_CHECK = True
    APPLY_STAIN_NORMALIZATION = False  # Set True if model trained with normalization

# ============================================================================
# LOGGING
# ============================================================================

os.makedirs('logs', exist_ok=True)

class ASCIIFormatter(logging.Formatter):
    def format(self, record):
        msg = super().format(record)
        return msg.encode('ascii', 'replace').decode('ascii')

file_handler = logging.FileHandler('logs/app.log', encoding='utf-8')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(ASCIIFormatter('%(asctime)s - %(levelname)s - %(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, Config.LOG_LEVEL))
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ============================================================================
# STAIN NORMALIZATION
# ============================================================================

class ReinhardNormalizer:
    TARGET_MEANS = np.array([148.60, 41.56, 192.27])
    TARGET_STDS = np.array([41.56, 15.57, 26.19])
    
    @staticmethod
    def normalize(img: np.ndarray) -> np.ndarray:
        try:
            img_lab = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2LAB)
            img_means = img_lab.mean(axis=(0, 1))
            img_stds = img_lab.std(axis=(0, 1))
            img_stds = np.where(img_stds == 0, 1, img_stds)
            img_lab_normalized = ((img_lab - img_means) / img_stds) * ReinhardNormalizer.TARGET_STDS + ReinhardNormalizer.TARGET_MEANS
            img_lab_normalized = np.clip(img_lab_normalized, 0, 255).astype(np.uint8)
            return cv2.cvtColor(img_lab_normalized, cv2.COLOR_LAB2RGB)
        except Exception as e:
            logger.warning(f"Stain normalization failed: {e}")
            return img

# ============================================================================
# GRAD-CAM - HANDLES NESTED SEQUENTIAL MODELS
# ============================================================================

class GradCAM:
    """Ultra-simple Grad-CAM using manual forward pass"""
    
    def __init__(self, model):
        self.model = model
        self.conv_layer_name = self._find_last_conv_layer()
        logger.info(f"✅ Grad-CAM ready: {self.conv_layer_name}")
    
    def _get_all_layers_flat(self, model, parent_name=""):
        """Get all layers flattened"""
        all_layers = []
        for layer in model.layers:
            full_name = f"{parent_name}/{layer.name}" if parent_name else layer.name
            if hasattr(layer, 'layers') and len(layer.layers) > 0:
                all_layers.extend(self._get_all_layers_flat(layer, full_name))
            else:
                all_layers.append((full_name, layer))
        return all_layers
    
    def _find_last_conv_layer(self):
        """Find last Conv layer"""
        logger.info("🔍 Finding Conv layers...")
        all_layers = self._get_all_layers_flat(self.model)
        conv_layers = [(name, layer) for name, layer in all_layers 
                       if 'Conv' in layer.__class__.__name__]
        
        if not conv_layers:
            raise ValueError("No Conv layers found")
        
        for name, _ in conv_layers:
            logger.info(f"  ✓ {name}")
        
        last_name = conv_layers[-1][0]
        logger.info(f"  🎯 Using: {last_name}")
        return last_name
    
    def generate_heatmap(self, img_array: np.ndarray) -> np.ndarray:
        """Generate Grad-CAM using manual forward pass"""
        try:
            logger.info("🔥 Generating Grad-CAM...")
            
            img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
            conv_output = None
            
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(img_tensor)
                x = img_tensor
                
                # Parse layer path
                if '/' in self.conv_layer_name:
                    parts = self.conv_layer_name.split('/')
                    parent_name, target_name = parts[0], parts[1]
                    
                    # Go through top-level layers
                    for layer in self.model.layers:
                        if layer.name == parent_name:
                            # Navigate nested layers
                            for sublayer in layer.layers:
                                x = sublayer(x, training=False)
                                if sublayer.name == target_name:
                                    conv_output = x
                        else:
                            # Other top-level layers
                            x = layer(x, training=False)
                else:
                    # Top-level layer
                    for layer in self.model.layers:
                        x = layer(x, training=False)
                        if layer.name == self.conv_layer_name:
                            conv_output = x
                
                final_output = x
                loss = final_output[:, 0] if final_output.shape[-1] == 1 else final_output[:, 0]
            
            if conv_output is None:
                raise ValueError(f"Could not capture {self.conv_layer_name}")
            
            grads = tape.gradient(loss, conv_output)
            if grads is None:
                raise ValueError("Gradients are None")
            
            # Compute heatmap
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            conv_output_np = conv_output[0].numpy()
            pooled_grads_np = pooled_grads.numpy()
            
            for i in range(len(pooled_grads_np)):
                conv_output_np[:, :, i] *= pooled_grads_np[i]
            
            heatmap = np.mean(conv_output_np, axis=-1)
            heatmap = np.maximum(heatmap, 0)
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            
            logger.info(f"  ✅ Heatmap: {heatmap.shape}, [{heatmap.min():.3f}, {heatmap.max():.3f}]")
            return heatmap
            
        except Exception as e:
            logger.error(f"❌ Grad-CAM failed: {e}", exc_info=True)
            raise
    
    def overlay_heatmap(self, heatmap: np.ndarray, img: np.ndarray, alpha: float = 0.4) -> np.ndarray:
        """Overlay heatmap on image"""
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(img, 1 - alpha, heatmap_colored, alpha, 0)

# ============================================================================
# IMAGE PROCESSING
# ============================================================================

class ImageValidator:
    @staticmethod
    def is_histopathology(img: Image.Image) -> Tuple[bool, str]:
        try:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img)
            
            if img_array.shape[0] < Config.MIN_IMAGE_SIZE or img_array.shape[1] < Config.MIN_IMAGE_SIZE:
                return False, f"Image too small (min: {Config.MIN_IMAGE_SIZE}x{Config.MIN_IMAGE_SIZE})"
            
            if np.std(img_array) < 5:
                return False, "Image appears blank"
            
            if np.var(img_array) < 50:
                return False, "Insufficient color variation"
            
            return True, "Valid"
        except Exception as e:
            return False, str(e)

class ImageProcessor:
    @staticmethod
    def validate_image(file) -> Tuple[bool, str]:
        if not file or file.filename == '':
            return False, "No file"
        ext = file.filename.rsplit('.', 1)[1].lower() if '.' in file.filename else ''
        if ext not in Config.ALLOWED_EXTENSIONS:
            return False, f"Invalid type (allowed: {', '.join(Config.ALLOWED_EXTENSIONS)})"
        return True, "Valid"
    
    @staticmethod
    def preprocess(image_bytes: bytes, apply_stain_norm: bool = None) -> Tuple[np.ndarray, np.ndarray, str, bool]:
        try:
            img = Image.open(io.BytesIO(image_bytes))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            if Config.HISTOLOGY_CHECK:
                is_valid, msg = ImageValidator.is_histopathology(img)
                if not is_valid:
                    raise ValueError(msg)
            
            original_array = np.array(img)
            should_normalize = apply_stain_norm if apply_stain_norm is not None else Config.APPLY_STAIN_NORMALIZATION
            
            if should_normalize:
                logger.info("Applying stain normalization")
                normalized_array = ReinhardNormalizer.normalize(original_array)
                stain_normalized = True
            else:
                logger.info("Skipping stain normalization")
                normalized_array = original_array
                stain_normalized = False
            
            img_resized = Image.fromarray(normalized_array).resize(Config.IMG_SIZE, Image.Resampling.LANCZOS)
            img_array = np.array(img_resized, dtype=np.float32) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            original_rgb = np.array(img_resized)
            
            return img_array, original_rgb, "Success", stain_normalized
        except Exception as e:
            raise ValueError(f"Preprocessing failed: {e}")

# ============================================================================
# MODEL MANAGER
# ============================================================================

class ModelManager:
    _instance = None
    _model = None
    _grad_cam = None
    _model_info = {}
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def load_model(self) -> bool:
        try:
            if not os.path.exists(Config.MODEL_PATH):
                logger.error(f"Model not found: {Config.MODEL_PATH}")
                return False
            
            logger.info(f"Loading model: {Config.MODEL_PATH}")
            self._model = tf.keras.models.load_model(Config.MODEL_PATH)
            
            logger.info("Model layers:")
            for i, layer in enumerate(self._model.layers):
                logger.info(f"  {i}: {layer.name} ({layer.__class__.__name__})")
            
            try:
                self._grad_cam = GradCAM(self._model)
            except Exception as e:
                logger.error(f"Grad-CAM init failed: {e}", exc_info=True)
                self._grad_cam = None
            
            self._model_info = {
                'name': 'Gastric Cancer Detection CNN',
                'version': '1.0.0',
                'input_shape': str(self._model.input_shape),
                'output_shape': str(self._model.output_shape),
                'total_params': int(self._model.count_params()),
                'grad_cam_available': self._grad_cam is not None,
                'grad_cam_layer': self._grad_cam.conv_layer_name if self._grad_cam else None,
                'stain_normalization_enabled': Config.APPLY_STAIN_NORMALIZATION,
                'metrics': {
                    'accuracy': 86.87, 'specificity': 93.75, 
                    'sensitivity': 80.39, 'auc_roc': 0.9464,
                    'threshold': Config.THRESHOLD
                }
            }
            
            logger.info("✅ Model loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Model loading failed: {e}", exc_info=True)
            return False
    
    def predict(self, image_array: np.ndarray, original_rgb: np.ndarray, 
                generate_gradcam: bool = True) -> Dict[str, Any]:
        if not self._model:
            raise RuntimeError("Model not loaded")
        
        try:
            pred_proba = self._model.predict(image_array, verbose=0)
            abnormal_prob = float(pred_proba[0][0])
            normal_prob = 1.0 - abnormal_prob
            prediction = 'Abnormal' if abnormal_prob > Config.THRESHOLD else 'Normal'
            confidence = abnormal_prob if abnormal_prob > Config.THRESHOLD else normal_prob
            
            result = {
                'prediction': prediction,
                'confidence': round(confidence * 100, 2),
                'probabilities': {
                    'normal': round(normal_prob * 100, 2),
                    'abnormal': round(abnormal_prob * 100, 2)
                },
                'raw_score': float(abnormal_prob),
                'threshold': Config.THRESHOLD
            }
            
            if generate_gradcam and self._grad_cam:
                try:
                    heatmap = self._grad_cam.generate_heatmap(image_array)
                    overlayed = self._grad_cam.overlay_heatmap(heatmap, original_rgb)
                    _, buffer = cv2.imencode('.png', cv2.cvtColor(overlayed, cv2.COLOR_RGB2BGR))
                    result['gradcam'] = f"data:image/png;base64,{base64.b64encode(buffer).decode()}"
                    result['gradcam_available'] = True
                except Exception as e:
                    logger.error(f"Grad-CAM generation failed: {e}")
                    result['gradcam_available'] = False
                    result['gradcam_error'] = str(e)
            else:
                result['gradcam_available'] = False
            
            return result
        except Exception as e:
            logger.error(f"Prediction failed: {e}", exc_info=True)
            raise
    
    @property
    def is_loaded(self) -> bool:
        return self._model is not None
    
    @property
    def info(self) -> Dict[str, Any]:
        return self._model_info

model_manager = ModelManager()

# ============================================================================
# FLASK APP
# ============================================================================

app = Flask(__name__, static_folder='static', static_url_path='')
app.config.from_object(Config)
CORS(app, resources={r"/api/*": {"origins": Config.CORS_ORIGINS}})

def require_model(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not model_manager.is_loaded:
            return jsonify({'success': False, 'error': 'Model not loaded'}), 503
        return f(*args, **kwargs)
    return decorated

def handle_errors(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            return jsonify({'success': False, 'error': 'Internal error'}), 500
    return decorated

@app.route('/')
def index():
    try:
        return send_from_directory('static', 'index.html')
    except:
        return jsonify({'message': 'API running', 'endpoints': {
            'health': '/api/health', 'model_info': '/api/model-info', 
            'predict': '/api/predict (POST)'
        }})

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

@app.route('/api/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_manager.is_loaded,
        'grad_cam_available': model_manager._grad_cam is not None if model_manager.is_loaded else False,
        'stain_normalization': Config.APPLY_STAIN_NORMALIZATION,
        'timestamp': datetime.utcnow().isoformat()
    })

@app.route('/api/model-info')
@require_model
def get_model_info():
    return jsonify({'success': True, 'data': model_manager.info})

@app.route('/api/predict', methods=['POST'])
@require_model
@handle_errors
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image file'}), 400
    
    file = request.files['image']
    is_valid, msg = ImageProcessor.validate_image(file)
    if not is_valid:
        return jsonify({'success': False, 'error': msg}), 400
    
    generate_gradcam = request.form.get('gradcam', 'true').lower() == 'true'
    apply_stain_norm = None
    if 'stain_normalize' in request.form:
        apply_stain_norm = request.form.get('stain_normalize').lower() == 'true'
    
    image_bytes = file.read()
    preprocessed, original_rgb, msg, stain_normalized = ImageProcessor.preprocess(
        image_bytes, apply_stain_norm
    )
    
    logger.info(f"Preprocessing: {msg}, Stain normalized: {stain_normalized}")
    result = model_manager.predict(preprocessed, original_rgb, generate_gradcam)
    logger.info(f"Prediction: {result['prediction']} ({result['confidence']}%)")
    
    return jsonify({
        'success': True,
        'data': {
            **result,
            'preprocessing': {
                'stain_normalized': stain_normalized,
                'resized_to': f"{Config.IMG_SIZE[0]}x{Config.IMG_SIZE[1]}",
                'validated': True
            },
            'timestamp': datetime.utcnow().isoformat(),
            'filename': secure_filename(file.filename)
        }
    })

@app.errorhandler(404)
def not_found(e):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({'success': False, 'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({'success': False, 'error': 'Internal error'}), 500

def initialize_app():
    logger.info("=" * 70)
    logger.info("GASTRIC CANCER DETECTION API")
    logger.info("=" * 70)
    logger.info(f"Model: {Config.MODEL_PATH}")
    logger.info(f"Input size: {Config.IMG_SIZE}")
    logger.info(f"Threshold: {Config.THRESHOLD}")
    logger.info(f"Stain normalization: {Config.APPLY_STAIN_NORMALIZATION}")
    logger.info("=" * 70)
    
    if not model_manager.load_model():
        logger.error("Failed to load model")
        return False
    
    logger.info("\n✅ Application ready")
    logger.info("Endpoints: /, /api/health, /api/model-info, /api/predict")
    logger.info("=" * 70)
    return True
# ============================================================================
# INITIALIZE MODEL ON STARTUP (for Gunicorn)
# ============================================================================

# This runs when the module is imported by Gunicorn
if not model_manager.is_loaded:
    logger.info("Initializing application for production deployment...")
    if not initialize_app():
        logger.critical("⚠️ Model initialization failed - API will return 503 errors")

# ============================================================================
# DEVELOPMENT SERVER
# ============================================================================

if __name__ == '__main__':
    if not model_manager.is_loaded:
        if not initialize_app():
            logger.error("Initialization failed")
            exit(1)
    
    host = os.getenv('HOST', '0.0.0.0')
    port = int(os.getenv('PORT', 5000))
    logger.info(f"\n🚀 Server: http://{host}:{port}\n")
    app.run(host=host, port=port, debug=Config.DEBUG, threaded=True)
        
