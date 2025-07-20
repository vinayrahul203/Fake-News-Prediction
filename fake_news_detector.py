import pandas as pd
import numpy as np
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========================
# 1. Load and Label Data
# ========================

def load_and_prepare_data():
    print("Loading datasets...")
    fake_df = pd.read_csv("Fake.csv", on_bad_lines='skip', engine='python')
    true_df = pd.read_csv("True.csv", on_bad_lines='skip', engine='python')
    
    fake_df['label'] = 0
    true_df['label'] = 1

    df = pd.concat([fake_df[['text', 'label']], true_df[['text', 'label']]], ignore_index=True)
    df.dropna(inplace=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Data loaded and combined.")
    return df

# ========================
# 2. Preprocessing
# ========================

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ========================
# 3. Vectorization
# ========================

def vectorize_text(data):
    print("Cleaning and vectorizing text...")
    data['text'] = data['text'].apply(clean_text)
    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X = vectorizer.fit_transform(data['text'])
    y = data['label']
    return X, y, vectorizer

# ========================
# 4. Train & Evaluate Models
# ========================

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("Training models...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "SVM (Linear)": LinearSVC()
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = (model, acc)
        print(f"{name}: Accuracy = {acc:.4f}")
    
    return results

# ========================
# 5. Save Best Model
# ========================

def save_best_model(results, vectorizer):
    best_model_name = max(results, key=lambda k: results[k][1])
    best_model, best_acc = results[best_model_name]
    print(f"\n✅ Best Model: {best_model_name} with Accuracy = {best_acc:.4f}")

    joblib.dump(best_model, "svm_model.pkl")
    joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
    print("Model and vectorizer saved as 'svm_model.pkl' and 'tfidf_vectorizer.pkl'.")

    return best_model

# ========================
# 6. Predict Function
# ========================

def predict_news(text):
    if not os.path.exists("svm_model.pkl") or not os.path.exists("tfidf_vectorizer.pkl"):
        print("❌ Trained model/vectorizer not found. Run training first.")
        return

    model = joblib.load("svm_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    result = "Real News ✅" if prediction == 1 else "Fake News ❌"
    print(f"Input: {text}\nPrediction: {result}")

# ========================
# 7. Confusion Matrix Plot
# ========================

def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

# ========================
# 🔁 Main Execution
# ========================

if __name__ == "__main__":
    # Load data
    data = load_and_prepare_data()

    # Vectorize
    X, y, vectorizer = vectorize_text(data)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Train and Evaluate
    results = train_and_evaluate_models(X_train, X_test, y_train, y_test)

    # Save Best Model
    best_model = save_best_model(results, vectorizer)

    # Confusion Matrix
    plot_confusion_matrix(best_model, X_test, y_test)

    # Predict Example
    print("\n🔍 Sample Predictions:")
    predict_news("NASA confirms water on the Moon.")
    predict_news("Donald Trump Sends Out Embarrassing New Year’s Eve Message.")
