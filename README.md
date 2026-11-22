# Phishing Email Detection Model

A machine learning model built with Scikit-learn that classifies emails as **Phishing** or **Safe** using both the text content and URL-based features of each email.

## Features

- Trains on a dataset of phishing and legitimate emails
- Extracts and analyzes email features (URLs, keywords, formatting, etc.)
- Classifies emails as "Phishing" or "Safe"
- Displays accuracy, a classification report, and a confusion matrix
- Saves the trained model so single emails can be tested later

## Dataset

This project uses the **Phishing Email Dataset** from Kaggle (columns: `Email Text`, `Email Type`).

Dataset link: *[add the Kaggle link you used here]*

Download the CSV and place it in the project folder (it is not included in this repository).

## How It Works

1. **Load and clean data:** remove missing values and duplicates, convert labels to 0 (Safe) and 1 (Phishing).
2. **Feature extraction:**
   - **TF-IDF** on the email text (unigrams and bigrams)
   - **Hand-crafted features:**
     - number of URLs
     - IP address used as a link
     - shortened links (bit.ly, tinyurl, etc.)
     - average URL length
     - `@` symbols inside URLs
     - count of suspicious keywords (urgent, verify, password, suspended, etc.)
     - exclamation mark count
     - uppercase letter ratio
     - presence of HTML
     - email length
3. **Train:** Logistic Regression with an 80/20 train/test split.
4. **Evaluate:** accuracy, precision, recall, F1-score, and confusion matrix.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Train and evaluate the model:

```bash
python phishing_detector.py --data Phishing_Email.csv
```

If your CSV uses different column names:

```bash
python phishing_detector.py --data emails.csv --text-col "Email Text" --label-col "Email Type"
```

Classify a single email using the saved model:

```bash
python phishing_detector.py --predict "Your account is suspended. Verify now at http://bit.ly/abc123"
```

Quick pipeline test with a tiny built-in sample:

```bash
python phishing_detector.py --demo
```

## Results

| Metric | Score |
|--------|-------|
| Accuracy | **XX.XX%** |
| Precision (Phishing) | XX.XX% |
| Recall (Phishing) | XX.XX% |
| F1-score (Phishing) | XX.XX% |

### Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

## Project Structure

```
.
├── phishing_detector.py    # training, evaluation, and prediction
├── requirements.txt        # dependencies
├── confusion_matrix.png    # generated after training
└── README.md
```

## Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy, SciPy
- Matplotlib

## Future Improvements

- Try other models (Random Forest, SVM, XGBoost) and compare results
- Add email header analysis (sender domain, SPF/DKIM)
- Build a simple web interface for testing emails
- Use a larger and more recent dataset

## Author

*[Your name]* – *[your GitHub profile link]*
