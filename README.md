# BERT News Classifier

This project features a complete end-to-end pipeline for classifying news articles into four categories: **World**, **Sports**, **Business**, and **Sci/Tech**. It includes a fine-tuned BERT model, a Flask-based backend API, and a modern, responsive web interface.

## 🚀 Features

* **Real-time Classification**: Instantly categorize news headlines or snippets using a fine-tuned BERT model.
* **Confidence Scoring**: View the probability distribution for all four categories.
* **Interactive Web UI**: A clean, user-friendly interface with auto-resizing text areas and keyboard shortcuts (Ctrl+Enter).
* **History Tracking**: Locally stored history of your previous classifications.
* **Dockerized Deployment**: Easily deploy the entire stack using Docker and Docker Compose.

## 🧠 Model & Dataset

The model is built on the `bert-base-uncased` architecture and fine-tuned on the **AG News Dataset**.

* **Dataset**: 120,000 training samples and 7,600 test samples.
* **Categories**: World (0), Sports (1), Business (2), Sci/Tech (3).
* **Input Handling**: Raw text is cleaned using regex to remove noise and normalized before tokenization.
* **Architecture**: Uses Hugging Face `transformers` for the tokenizer and TensorFlow for the SavedModel inference.

## 🛠️ Technology Stack

* **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
* **Backend**: Flask (Python 3.10)
* **Machine Learning**: TensorFlow 2.16.1, Hugging Face Transformers 4.42.3
* **Deployment**: Docker, Gunicorn

## 📦 Installation & Setup

### Prerequisites

* Docker and Docker Compose
* The `saved_models/` directory (containing your fine-tuned BERT model and tokenizer)
* model link : https://drive.google.com/file/d/1d0A2kV9aRi9LnRLv7tA2Ed_M-VO8zkPT/view?usp=sharing

### Running with Docker (Recommended)

1. Ensure your model files are located in `./saved_models/bert_sentiment_...` and your tokenizer in `./saved_models/tokenizer`.
2. Run the following command:
```bash
docker-compose up --build

```


3. Access the application at `http://localhost:8000`.

### Manual Setup

1. **Install Dependencies**:
```bash
pip install -r requirements.txt

```


2. **Start the Flask App**:
```bash
python app.py

```


3. The app will be available at `http://localhost:5000`.

## 📡 API Endpoints

* `GET /`: Home page.
* `POST /predict`: Receives JSON `{"text": "..."}` and returns the predicted category, confidence, and probability distribution.
* `GET /health`: Returns the status of the model and tokenizer.
* `GET /model-info`: Provides metadata about the loaded BERT model.

