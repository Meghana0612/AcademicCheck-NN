import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.similarity import build_feature_matrix

# Load saved artifacts (all stored in one dict inside model.pkl)
artifacts = pickle.load(open("models/model.pkl", "rb"))
model          = artifacts["model"]
scaler         = artifacts["scaler"]
tfidf_vectorizer = artifacts["tfidf_vectorizer"]

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Compute features using correct function
features_df, _ = build_feature_matrix(df, tfidf_vectorizer=tfidf_vectorizer)

# Scale features
X = scaler.transform(features_df)
y = df["label"]

# Predict
y_pred = model.predict(X)

# Results
print(f"\n✅ Accuracy      : {accuracy_score(y, y_pred) * 100:.2f}%")
print(f"\n📊 Classification Report:\n")
print(classification_report(y, y_pred, target_names=["Not Plagiarized", "Plagiarized"]))
print(f"\n🔢 Confusion Matrix:\n")
print(confusion_matrix(y, y_pred))