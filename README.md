🧑‍💼 Personality Prediction System Through CV Analysis

This project predicts a candidate’s personality traits (OCEAN – Openness, Conscientiousness, Extraversion, Agreeableness, Neuroticism) by analyzing their resume (PDF/DOCX).
It combines Natural Language Processing (NLP), Machine Learning (Logistic Regression), and Flask to provide an interactive web application.

🚀 Features

Upload resumes in PDF or DOCX format.

Handles scanned PDFs using OCR (Tesseract).

Extracts text from resumes using PyPDF2, docx2txt, and pytesseract.

Uses TF-IDF Vectorization + Logistic Regression trained on the Essays Big5 dataset
.

Predicts OCEAN traits as High or Low.

Web UI built with Flask + HTML + CSS.                                        

📊 Example Output

O (Openness): High → Creative, curious, imaginative

C (Conscientiousness): Low → Spontaneous, less structured

E (Extraversion): High → Outgoing, talkative

A (Agreeableness): Low → Competitive, critical

N (Neuroticism): High → Sensitive, anxious

🛠️ Tech Stack

Backend: Flask

ML Models: Logistic Regression (scikit-learn)

NLP: TF-IDF Vectorizer

Dataset: Essays Big5

File Handling: PyPDF2, docx2txt, pdf2image, pytesseract 


⚙️ Installation

1. Clone the repository

git clone https://github.com/your-username/personality-prediction-cv.git
cd personality-prediction-cv


2. Create and activate a virtual environment

python -m venv venv
venv\Scripts\activate   # On Windows
source venv/bin/activate   # On Linux/Mac


3. Install dependencies

pip install -r requirements.txt


4. Install Tesseract OCR (for scanned PDFs)

Windows: Download here

Linux: sudo apt install tesseract-ocr

Mac: brew install tesseract

Update the path in resume_parser.py if needed:

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

🏋️ Training Models

Run the training script to generate models:

python train_model.py


This will create the models/ folder containing TF-IDF vectorizer and trained classifiers.

▶️ Running the App

Start the Flask server:

python app.py


Visit http://127.0.0.1:5000/
 in your browser.
