import os
import tensorflow as tf
from transformers import AutoTokenizer
from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime

# --- Configuration ---
SAVE_DIR = "saved_models" 
MODEL_PATH = os.path.join(SAVE_DIR, "bert_sentiment_20251206_162615")
TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer")

# The class names for the AG_News dataset
CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
MAX_LENGTH = 128

# Global variables for model and tokenizer
tokenizer = None
infer = None
NUM_CLASSES = 0

def load_model():
    """Loads the model and tokenizer globally."""
    global tokenizer, infer, NUM_CLASSES
    print("🤖 Loading tokenizer...")
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
        print("✅ Tokenizer loaded.")
        
        print("🧠 Loading TensorFlow SavedModel...")
        # Load the SavedModel
        loaded_model = tf.saved_model.load(MODEL_PATH)
        # Assuming 'serving_default' is the correct signature for prediction
        infer = loaded_model.signatures['serving_default']
        print("✅ Model loaded.")
        
        # Determine number of classes from the model's output structure
        output_name = list(infer.structured_outputs.keys())[0]
        NUM_CLASSES = infer.structured_outputs[output_name].shape[1]
        print(f"Model outputs {NUM_CLASSES} classes.")

    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        raise RuntimeError(f"Could not load required components: {e}")

# --- Flask App Initialization ---
app = Flask(__name__)

# Load the model when the Flask application starts
with app.app_context():
    load_model()

# --- Routes ---
@app.route('/')
def home():
    """Renders the main HTML page."""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_predict():
    """Handles the POST request for prediction and returns JSON."""
    if request.is_json:
        data = request.get_json()
        text = data.get('text', '')
    else:
        text = request.form.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided for classification.'}), 400

    try:
        # Input validation
        if len(text) > 2000:
            return jsonify({'error': 'Text too long. Maximum 2000 characters allowed.'}), 400
        
        # Tokenize the input text
        inputs = tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors='np'
        )
        
        # Create input dictionary
        input_dict = {
            'input_ids': tf.constant(inputs['input_ids'], dtype=tf.int64),
            'attention_mask': tf.constant(inputs['attention_mask'], dtype=tf.int64),
            'token_type_ids': tf.constant(inputs['token_type_ids'], dtype=tf.int64)
        }
        
        # Run the prediction
        output = infer(**input_dict)
        
        # Extract predictions
        output_key = list(output.keys())[0]
        predictions = output[output_key]
        
        # Apply softmax to get probabilities
        probabilities = tf.nn.softmax(predictions, axis=-1).numpy()[0]
        
        # Get the index of the highest probability
        predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]
        
        # Get confidence score
        confidence = float(probabilities[predicted_class_index])
        
        # Get category name
        if NUM_CLASSES == 4 and len(CLASS_NAMES) == 4:
            category = CLASS_NAMES[predicted_class_index]
        else:
            category = f"Class {predicted_class_index}"

        # Prepare detailed response
        response = {
            'success': True,
            'category': category,
            'confidence': confidence,
            'all_probabilities': probabilities.tolist(),
            'predicted_index': int(predicted_class_index),
            'text_preview': text[:100] + ('...' if len(text) > 100 else ''),
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_name': 'BERT News Classifier',
                'num_classes': NUM_CLASSES,
                'max_length': MAX_LENGTH
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        app.logger.error(f"Prediction failed: {str(e)}")
        return jsonify({'error': f'An error occurred during prediction: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for monitoring."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': infer is not None,
        'tokenizer_loaded': tokenizer is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info', methods=['GET'])
def model_info():
    """Returns information about the loaded model."""
    return jsonify({
        'model_name': 'BERT News Classifier',
        'classes': CLASS_NAMES,
        'max_length': MAX_LENGTH,
        'num_classes': NUM_CLASSES,
        'model_path': MODEL_PATH
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)