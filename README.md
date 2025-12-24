# BERT-Based News Text Classification

**Fine-Tuning BERT on the AG News Dataset and Deploying a Real-Time Text Classification System**

---

## Project Overview

This project demonstrates how to **fine-tune a pre-trained BERT model** for text classification and deploy it as a **real-time web application**. It is designed to classify news articles into **4 categories**:  

- World  
- Sports  
- Business  
- Science & Technology  

The system is fully **containerized with Docker** and uses **Flask** for the backend, enabling scalable and reproducible deployment.

---

## Objectives

- Build a high-accuracy text classification model using BERT.  
- Fine-tune a pre-trained BERT model on the AG News dataset.  
- Deploy the model in a web application for **real-time inference**.  
- Ensure reproducibility and scalability using Docker.

---

## Model & Architecture

**BERT (Bidirectional Encoder Representations from Transformers)**  
- Pre-trained on massive corpora (Wikipedia, BooksCorpus).  
- Uses self-attention for contextual understanding of text.  
- Fine-tuned for classification by adding a task-specific output layer.  

**Why Transformers/BERT?**  
- Handles context in both directions.  
- Achieves **state-of-the-art performance** on NLP tasks.  
- Requires less data and converges faster compared to training from scratch.

---

## Dataset & Training

**Dataset:** AG News  
- 4 balanced categories: World, Sports, Business, Sci/Tech  
- 120,000 training samples, widely used benchmark dataset  

**Training Pipeline:**  
1. Text cleaning and preprocessing  
2. Tokenization using BERT tokenizer  
3. Input formatting (attention masks, padding)  
4. Fine-tuning the model  
5. Evaluation on the validation set  

**Results:**  
- Achieved **87.89% accuracy** on validation data.  
- Confusion matrix confirms strong generalization and minimal overfitting.

---

## System Architecture

**Components:**  
- **Frontend:** HTML/CSS user interface for text input.  
- **Backend:** Flask application handling requests and inference.  
- **Model:** Fine-tuned BERT with tokenizer.  
- **Deployment:** Docker + Gunicorn for scalable and reproducible setup.  

**Prediction Flow:**  
1. User enters text in web interface.  
2. Flask receives the request.  
3. Text is tokenized and fed into BERT.  
4. Predicted category and confidence scores are returned in real-time.

---

## Installation & Usage

**Clone the repository:**  
```bash
git clone https://github.com/Praydos/ai-projects-chafik-bouchti-atif
cd ai-projects-chafik-bouchti-atif
````

**Build and run with Docker:**

```bash
docker build -t bert-news-classifier .
docker run -p 8000:8000 bert-news-classifier
```

**Access the web application:**
Open your browser and go to `http://localhost:8000`.

---

## 📈 Demo

The web application allows users to input any news text and get **real-time classification** with confidence scores for each category.

<img width="668" height="314" alt="image" src="https://github.com/user-attachments/assets/e017b84c-9e15-4e27-9b10-ae3201d03867" />

<img width="532" height="333" alt="image" src="https://github.com/user-attachments/assets/6be95271-f810-4fba-b3ff-025484eef74c" />

<img width="598" height="298" alt="image" src="https://github.com/user-attachments/assets/14b3b29c-8522-454f-9902-f1759dc479d6" />

