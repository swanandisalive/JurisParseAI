import os
import json
import joblib
import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MultiLabelBinarizer

def train_and_export_pipeline():
    print("Starting JurisParse AI Offline Training Pipeline (Multi-Label)...")
    
    ARTIFACTS_DIR = "model_artifacts"
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    
    print("Fetching LEDGAR dataset from Hugging Face (coastalcph/lex_glue)...")
    dataset = load_dataset("coastalcph/lex_glue", "ledgar")
    
    label_names = dataset["train"].features["label"].names
    df_train = pd.DataFrame(dataset["train"])
    df_val = pd.DataFrame(dataset["validation"])
    
    df_train["label_name"] = df_train["label"].apply(lambda x: label_names[x])
    df_val["label_name"] = df_val["label"].apply(lambda x: label_names[x])
    
    top_5_classes = df_train["label_name"].value_counts().head(5).index.tolist()
    print(f"Selected Top 5 Target Clause Categories:\n   {top_5_classes}")
    
    df_train_filtered = df_train[df_train["label_name"].isin(top_5_classes)].copy()
    df_val_filtered = df_val[df_val["label_name"].isin(top_5_classes)].copy()
    
    MAX_SAMPLES_PER_CLASS = 1000
    df_train_sampled = (
        df_train_filtered.groupby("label_name")
        .apply(lambda x: x.sample(min(len(x), MAX_SAMPLES_PER_CLASS), random_state=42))
        .reset_index(drop=True)
    )
    
    # 1. UPGRADE: Wrap single labels in lists for MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    y_train = mlb.fit_transform([[label] for label in df_train_sampled["label_name"]])
    y_val = mlb.transform([[label] for label in df_val_filtered["label_name"]])
    
    X_train_text = df_train_sampled["text"]
    X_val_text = df_val_filtered["text"]
    
    print("Extracting TF-IDF Features...")
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), stop_words="english", sublinear_tf=True)
    X_train_tfidf = tfidf.fit_transform(X_train_text)
    X_val_tfidf = tfidf.transform(X_val_text)
    
    # 2. UPGRADE: One-Vs-Rest with Balanced Class Weights
    print("Training Multi-Label Logistic Regression Classifier...")
    base_clf = LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", class_weight="balanced")
    clf = OneVsRestClassifier(base_clf)
    clf.fit(X_train_tfidf, y_train)
    
    print("Evaluating model performance...")
    y_pred = clf.predict(X_val_tfidf)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred, target_names=mlb.classes_, output_dict=True, zero_division=0)
    
    print(f"Exact Match Accuracy: {acc * 100:.2f}%")
    
    # 3. Save updated artifacts (Note: label_encoder is now mlb_encoder)
    print("Saving pre-trained model artifacts...")
    joblib.dump(clf, os.path.join(ARTIFACTS_DIR, "clause_classifier.pkl"))
    joblib.dump(tfidf, os.path.join(ARTIFACTS_DIR, "tfidf_vectorizer.pkl"))
    joblib.dump(mlb, os.path.join(ARTIFACTS_DIR, "mlb_encoder.pkl"))
    
    # Extract top keywords
    feature_names = np.array(tfidf.get_feature_names_out())
    top_keywords_per_class = {}
    for i, class_label in enumerate(mlb.classes_):
        top_indices = np.argsort(clf.estimators_[i].coef_[0])[-10:][::-1]
        top_keywords_per_class[class_label] = feature_names[top_indices].tolist()

    metrics_payload = {"accuracy": acc, "classes": mlb.classes_.tolist(), "classification_report": report}
    with open(os.path.join(ARTIFACTS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=4)
        
    with open(os.path.join(ARTIFACTS_DIR, "top_keywords.json"), "w") as f:
        json.dump(top_keywords_per_class, f, indent=4)
        
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    train_and_export_pipeline()