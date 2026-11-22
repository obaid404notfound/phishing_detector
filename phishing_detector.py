"""
Phishing Email Detection Model (Scikit-learn)

Usage:
    python phishing_detector.py --data Phishing_Email.csv
    python phishing_detector.py --data emails.csv --text-col "Email Text" --label-col "Email Type"
    python phishing_detector.py --demo            # tiny built-in sample, just to test the pipeline
    python phishing_detector.py --predict "Your account is locked, verify at http://bit.ly/x1"

Dataset: the Kaggle "Phishing Email Dataset" works well (columns: "Email Text", "Email Type").
Any CSV with a text column and a label column will work.
"""
import argparse
import re

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

MODEL_PATH = "phishing_model.joblib"
URL_RE = re.compile(r"(https?://[^\s<>\"']+|www\.[^\s<>\"']+)", re.I)
IP_URL_RE = re.compile(r"https?://\d{1,3}(?:\.\d{1,3}){3}")
SHORTENERS = ("bit.ly", "tinyurl", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly")
KEYWORDS = ["urgent", "verify", "account", "password", "login", "suspended",
            "click", "confirm", "bank", "security", "update", "limited",
            "winner", "prize", "free", "immediately", "alert", "invoice"]
FEATURE_NAMES = ["num_urls", "has_ip_url", "has_shortener", "avg_url_len",
                 "num_at_in_urls", "keyword_count", "exclaim_count",
                 "upper_ratio", "has_html", "text_len"]


def extract_features(text: str) -> list:
    """Hand-crafted URL + keyword features for a single email."""
    text = str(text)
    low = text.lower()
    urls = URL_RE.findall(text)
    return [
        len(urls),
        int(bool(IP_URL_RE.search(text))),
        int(any(s in u.lower() for u in urls for s in SHORTENERS)),
        float(np.mean([len(u) for u in urls])) if urls else 0.0,
        sum(u.count("@") for u in urls),
        sum(low.count(k) for k in KEYWORDS),
        text.count("!"),
        sum(c.isupper() for c in text) / max(len(text), 1),
        int(bool(re.search(r"<\s*(html|a |form|body)", low))),
        len(text),
    ]


def load_data(args):
    if args.demo:
        phish = [
            "URGENT: Your account has been suspended. Verify your password now at http://192.168.4.7/login",
            "Dear customer, confirm your bank details immediately: http://bit.ly/3xYz update required!!!",
            "You are a WINNER! Claim your free prize, click here http://free-prize.example.com/claim",
            "Security alert: unusual login detected. Click to verify http://secure-paypa1.example.net",
            "Your invoice is overdue. Login to avoid account limitation www.pay-now-fast.example.org",
            "Final notice! Update your password immediately or your account will be closed http://tinyurl.com/a1",
        ] * 5
        safe = [
            "Hi team, the meeting is moved to 3pm tomorrow. Agenda attached.",
            "Thanks for your help yesterday, let's catch up for lunch this week.",
            "Please find the project report attached. Let me know your feedback.",
            "Reminder: the quarterly review is on Friday in conference room B.",
            "Happy birthday! Hope you have a great day with family and friends.",
            "Here are the notes from today's call. Next steps are listed below.",
        ] * 5
        return pd.DataFrame({"text": phish + safe, "label": [1] * len(phish) + [0] * len(safe)})

    df = pd.read_csv(args.data)
    df = df[[args.text_col, args.label_col]].dropna()
    df.columns = ["text", "label"]
    if df["label"].dtype == object:
        df["label"] = df["label"].str.lower().str.contains("phish").astype(int)
    else:
        df["label"] = df["label"].astype(int)  # assumes 1 = phishing
    return df.drop_duplicates().reset_index(drop=True)


def build_matrix(texts, tfidf, scaler, fit=False):
    feats = np.array([extract_features(t) for t in texts], dtype=float)
    if fit:
        tfidf_mat = tfidf.fit_transform(texts)
        feats = scaler.fit_transform(feats)
    else:
        tfidf_mat = tfidf.transform(texts)
        feats = scaler.transform(feats)
    return hstack([tfidf_mat, csr_matrix(feats)]).tocsr()


def train(args):
    df = load_data(args)
    print(f"Loaded {len(df)} emails | phishing: {df.label.sum()} | safe: {(df.label == 0).sum()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"])

    tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                            stop_words="english", sublinear_tf=True)
    scaler = StandardScaler()
    Xtr = build_matrix(X_train.tolist(), tfidf, scaler, fit=True)
    Xte = build_matrix(X_test.tolist(), tfidf, scaler)

    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(Xtr, y_train)
    pred = clf.predict(Xte)

    print(f"\nAccuracy: {accuracy_score(y_test, pred):.4f}\n")
    print(classification_report(y_test, pred, target_names=["Safe", "Phishing"]))

    cm = confusion_matrix(y_test, pred)
    print("Confusion matrix (rows = actual, cols = predicted):\n", cm)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Phishing"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Phishing Email Detection - Confusion Matrix")
    plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
    print("Saved confusion_matrix.png")

    joblib.dump({"tfidf": tfidf, "scaler": scaler, "clf": clf}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}")


def predict(text: str):
    m = joblib.load(MODEL_PATH)
    X = build_matrix([text], m["tfidf"], m["scaler"])
    p = m["clf"].predict_proba(X)[0, 1]
    print(f"{'Phishing' if p >= 0.5 else 'Safe'} (phishing probability: {p:.2%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="path to CSV dataset")
    ap.add_argument("--text-col", default="Email Text")
    ap.add_argument("--label-col", default="Email Type")
    ap.add_argument("--demo", action="store_true", help="use tiny built-in sample data")
    ap.add_argument("--predict", help="classify a single email text using the saved model")
    a = ap.parse_args()

    if a.predict:
        predict(a.predict)
    elif a.demo or a.data:
        train(a)
    else:
        ap.error("provide --data FILE.csv, --demo, or --predict TEXT")
