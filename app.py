# app.py
import os
import tensorflow as tf
from transformers import AutoTokenizer
from flask import Flask, render_template, request, jsonify

# --- Configuration ---
# Set the directory where your model and tokenizer are saved
# Make sure these paths match the ones copied into the Docker image
SAVE_DIR = "saved_models" 
MODEL_PATH = os.path.join(SAVE_DIR, "bert_sentiment_20251206_162615")
TOKENIZER_PATH = os.path.join(SAVE_DIR, "tokenizer")

# The class names for the AG_News dataset
# 0: World, 1: Sports, 2: Business, 3: Sci/Tech
CLASS_NAMES = ['World', 'Sports', 'Business', 'Sci/Tech']
MAX_LENGTH = 128

# --- Model Loading ---
# Global variables to hold the loaded model and tokenizer
# This ensures they are loaded only once when the app starts
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
        # The shape is usually [None, Num_Classes]
        NUM_CLASSES = infer.structured_outputs[output_name].shape[1]
        print(f"Model outputs {NUM_CLASSES} classes.")

    except Exception as e:
        print(f"❌ Error loading model or tokenizer: {e}")
        # In a real application, you might raise an exception or exit here
        raise RuntimeError(f"Could not load required components: {e}")

# --- Prediction Function ---
def predict(text):
    """Predicts the category of the input text."""
    if not infer:
        raise RuntimeError("Model is not loaded.")
        
    # Tokenize the input text
    inputs = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors='np' # Returns numpy arrays
    )
    
    # Create input dictionary using tf.constant and correct dtype (tf.int64)
    input_dict = {
        'input_ids': tf.constant(inputs['input_ids'], dtype=tf.int64),
        'attention_mask': tf.constant(inputs['attention_mask'], dtype=tf.int64),
        'token_type_ids': tf.constant(inputs['token_type_ids'], dtype=tf.int64)
    }
    
    # Run the prediction
    output = infer(**input_dict)
    
    # Extract predictions (e.g., 'output_0' or whatever the key is)
    output_key = list(output.keys())[0]
    predictions = output[output_key]
    
    # Process the output
    # Apply softmax to get probabilities
    probabilities = tf.nn.softmax(predictions, axis=-1).numpy()[0]
    # Get the index of the highest probability
    predicted_class_index = tf.argmax(predictions, axis=1).numpy()[0]

    # Map the index to a class name
    if NUM_CLASSES == 4 and len(CLASS_NAMES) == 4:
        sentiment = CLASS_NAMES[predicted_class_index]
    else:
        # Fallback or generic handling if class count doesn't match
        sentiment = f"Class {predicted_class_index}"

    return {
        'text': text,
        'category': sentiment,
        'predicted_class_index': int(predicted_class_index),
        'confidence': float(probabilities[predicted_class_index]),
        'probabilities': probabilities.tolist()
    }

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
        result = predict(text)
        # Return a simple JSON response for the frontend
        return jsonify({
            'success': True,
            'category': result['category'],
            'confidence': f"{result['confidence']:.2f}",
            'all_probabilities': result['probabilities']
        })
    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': f'An error occurred during prediction: {e}'}), 500

if __name__ == '__main__':
    # Start the Flask development server (only used without Docker in development)
    # The Dockerfile/docker-compose will use gunicorn for production deployment
    app.run(host='0.0.0.0', port=5000)