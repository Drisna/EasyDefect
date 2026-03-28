from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import numpy as np
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras import layers, models
import pickle
import glob

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_FOLDER = 'models'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MODEL_FOLDER'] = MODEL_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODEL_FOLDER, exist_ok=True)

# Global variable to store current model
current_model = None
current_model_name = None
model_threshold = 0.0054  # Default threshold from training

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def build_autoencoder(input_shape=(128, 128, 3)):
    """Build the autoencoder model"""
    # Encoder
    input_img = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
    
    # Decoder
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    autoencoder = models.Model(input_img, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def load_images_from_folder(folder_path, target_size=(128, 128)):
    """Load and preprocess images from folder"""
    images = []
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            try:
                img = tf.keras.preprocessing.image.load_img(
                    img_path, target_size=target_size
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0  # Normalize
                images.append(img_array)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
    return np.array(images)

def calculate_reconstruction_errors(model, images):
    """Calculate reconstruction errors for images"""
    reconstructed = model.predict(images, verbose=0)
    mse = np.mean(np.square(images - reconstructed), axis=(1, 2, 3))
    return mse

@app.route('/api/train/', methods=['POST', 'OPTIONS'])
def train_model():
    """Train the autoencoder model"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        data = request.get_json()
        model_name = data.get('model_name', 'autoencoder_model')
        
        # Load images from uploads folder
        print("Loading and extracting features from: uploads")
        images = load_images_from_folder('uploads')
        
        if len(images) == 0:
            return jsonify({'error': 'No images found in uploads folder'}), 400
            
        print(f"Loaded {len(images)} images.")
        print("Training Autoencoder...")
        
        # Build and train autoencoder
        autoencoder = build_autoencoder()
        
        # Train the model
        history = autoencoder.fit(
            images, images,
            epochs=50,
            batch_size=16,
            shuffle=True,
            validation_split=0.1,
            verbose=0
        )
        
        # Calculate reconstruction errors to set threshold
        errors = calculate_reconstruction_errors(autoencoder, images)
        threshold = np.percentile(errors, 95)  # 95th percentile as threshold
        
        # Save the model and threshold
        model_path = os.path.join(MODEL_FOLDER, f"{model_name}.h5")
        autoencoder.save(model_path)
        
        # Save threshold
        threshold_path = os.path.join(MODEL_FOLDER, f"{model_name}_threshold.pkl")
        with open(threshold_path, 'wb') as f:
            pickle.dump({'threshold': threshold}, f)
        
        # Update current model
        global current_model, current_model_name, model_threshold
        current_model = autoencoder
        current_model_name = model_name
        model_threshold = threshold
        
        print(f"Threshold set at {threshold:.4f} (percentile: 95)")
        print(f"Training complete. Model saved to: {model_path}")
        
        return jsonify({
            'message': 'Model trained successfully',
            'model_name': model_name,
            'threshold': float(threshold),
            'images_used': len(images),
            'final_loss': float(history.history['loss'][-1])
        })
        
    except Exception as e:
        print(f"Error in training: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/', methods=['POST', 'OPTIONS'])
def predict():
    """Predict if uploaded images are defective"""
    if request.method == 'OPTIONS':
        return '', 200
        
    try:
        # Check if model is loaded
        global current_model, current_model_name, model_threshold
        
        if current_model is None:
            # Try to load the latest model
            model_files = glob.glob(os.path.join(MODEL_FOLDER, "*.h5"))
            if not model_files:
                return jsonify({'error': 'No trained model found. Please train a model first.'}), 400
            
            # Load the most recent model
            latest_model = max(model_files, key=os.path.getctime)
            model_name = os.path.splitext(os.path.basename(latest_model))[0]
            
            # Load model
            current_model = tf.keras.models.load_model(latest_model)
            current_model_name = model_name
            
            # Load threshold
            threshold_path = os.path.join(MODEL_FOLDER, f"{model_name}_threshold.pkl")
            if os.path.exists(threshold_path):
                with open(threshold_path, 'rb') as f:
                    data = pickle.load(f)
                    model_threshold = data['threshold']
        
        # Get files from request
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        results = []
        normal_count = 0
        defective_count = 0
        
        for file in files:
            if file and allowed_file(file.filename):
                # Save temporarily
                filename = secure_filename(file.filename)
                temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{filename}")
                file.save(temp_path)
                
                # Load and preprocess image
                img = tf.keras.preprocessing.image.load_img(
                    temp_path, target_size=(128, 128)
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = img_array / 255.0
                img_array = np.expand_dims(img_array, axis=0)
                
                # Calculate reconstruction error
                reconstructed = current_model.predict(img_array, verbose=0)
                error = np.mean(np.square(img_array - reconstructed))
                
                # Classify based on threshold
                is_defective = error > model_threshold
                
                if is_defective:
                    defective_count += 1
                else:
                    normal_count += 1
                
                results.append({
                    'filename': filename,
                    'prediction': 'defective' if is_defective else 'normal',
                    'error': float(error),
                    'threshold': float(model_threshold)
                })
                
                # Clean up temp file
                os.remove(temp_path)
        
        return jsonify({
            'results': results,
            'summary': {
                'total': len(results),
                'normal': normal_count,
                'defective': defective_count,
                'normal_accuracy': f"{normal_count}/{len(results)}" if results else "0/0",
                'defective_accuracy': f"{defective_count}/{len(results)}" if results else "0/0"
            },
            'model_used': current_model_name,
            'threshold': float(model_threshold)
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/models/', methods=['GET'])
def list_models():
    """List all available trained models"""
    try:
        model_files = glob.glob(os.path.join(MODEL_FOLDER, "*.h5"))
        models_list = []
        
        for model_file in model_files:
            model_name = os.path.splitext(os.path.basename(model_file))[0]
            threshold_path = os.path.join(MODEL_FOLDER, f"{model_name}_threshold.pkl")
            
            model_info = {
                'name': model_name,
                'path': model_file,
                'created': os.path.getctime(model_file),
                'has_threshold': os.path.exists(threshold_path)
            }
            
            if os.path.exists(threshold_path):
                with open(threshold_path, 'rb') as f:
                    data = pickle.load(f)
                    model_info['threshold'] = float(data['threshold'])
            
            models_list.append(model_info)
        
        # Sort by creation date (newest first)
        models_list.sort(key=lambda x: x['created'], reverse=True)
        
        return jsonify({
            'models': models_list,
            'current_model': current_model_name
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/load-model/', methods=['POST'])
def load_model():
    """Load a specific trained model"""
    try:
        data = request.get_json()
        model_name = data.get('model_name')
        
        if not model_name:
            return jsonify({'error': 'Model name required'}), 400
        
        model_path = os.path.join(MODEL_FOLDER, f"{model_name}.h5")
        threshold_path = os.path.join(MODEL_FOLDER, f"{model_name}_threshold.pkl")
        
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 404
        
        # Load model
        global current_model, current_model_name, model_threshold
        current_model = tf.keras.models.load_model(model_path)
        current_model_name = model_name
        
        # Load threshold
        if os.path.exists(threshold_path):
            with open(threshold_path, 'rb') as f:
                data = pickle.load(f)
                model_threshold = data['threshold']
        
        return jsonify({
            'message': f'Model {model_name} loaded successfully',
            'model_name': model_name,
            'threshold': float(model_threshold)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)