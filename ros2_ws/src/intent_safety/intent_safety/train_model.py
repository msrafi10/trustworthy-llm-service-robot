import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import joblib
import os

def train():
    base_path = os.path.dirname(__file__)
    data_path = os.path.join(base_path, "intent_dataset.csv")

    print("Loading dataset...")
    data = pd.read_csv(data_path)

    X = data["text"]
    y = data["label"]

    print("Vectorizing text...")
    vectorizer = TfidfVectorizer()
    X_vec = vectorizer.fit_transform(X)

    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    print("Saving model...")
    joblib.dump(model, os.path.join(base_path, "intent_model.pkl"))
    joblib.dump(vectorizer, os.path.join(base_path, "vectorizer.pkl"))

    print("Training complete ✅")

if __name__ == "__main__":
    train()
