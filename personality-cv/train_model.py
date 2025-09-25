# train_model.py
import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from datasets import load_dataset

print(">> Loading Essays Big5 dataset from HuggingFace...")
dataset = load_dataset("jingjietan/essays-big5")

# Convert to DataFrame
df = dataset["train"].to_pandas()

print("Available columns:", df.columns.tolist())

# Use the correct column names: O, C, E, A, N
trait_columns = ["O", "C", "E", "A", "N"]

# Convert traits to integers
y_data = df[trait_columns].astype(int)
X_data = df["text"].astype(str)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

# TF-IDF vectorization
vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Save vectorizer
os.makedirs("models", exist_ok=True)
joblib.dump(vectorizer, "models/tfidf_vectorizer.joblib")

# Train one model per trait
for trait in trait_columns:
    print(f"\n>> Training model for {trait} ...")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_tfidf, y_train[trait])
    y_pred = clf.predict(X_test_tfidf)
    print(f"Classification report for trait {trait}:\n", classification_report(y_test[trait], y_pred))
    joblib.dump(clf, f"models/{trait}_clf.joblib")

print("\n✅ Training complete. Models and vectorizer saved in models/")
