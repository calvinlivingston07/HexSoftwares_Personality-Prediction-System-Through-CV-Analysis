import os
from flask import Flask, render_template, request
import joblib
from resume_parser import extract_text

app = Flask(__name__)

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load vectorizer
vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")
vectorizer = joblib.load(vectorizer_path)

# Load models (corrected _clf filenames)
traits = ["O", "C", "E", "A", "N"]
models = {}
for trait in traits:
    model_path = os.path.join(BASE_DIR, "models", f"{trait}_clf.joblib")
    print(f"Looking for model: {model_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    models[trait] = joblib.load(model_path)

@app.route("/", methods=["GET", "POST"])
def index():
    result = {}
    extracted_text = ""
    error_message = ""

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            try:
                extracted_text = extract_text(file)
                if not extracted_text.strip():
                    error_message = "No text could be extracted from the file."
                else:
                    # Transform text
                    X = vectorizer.transform([extracted_text])
                    # Predict traits
                    for trait in traits:
                        pred = models[trait].predict(X)[0]
                        result[trait] = "High" if pred == 1 else "Low"
            except Exception as e:
                error_message = f"Error reading file: {str(e)}"
        else:
            error_message = "No file uploaded."

    return render_template(
        "index.html",
        result=result,
        extracted_text=extracted_text.replace("\n", "<br>"),
        error_message=error_message
    )

if __name__ == "__main__":
    app.run(debug=True)
