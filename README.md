# Fake-News-Prediction
=========================================
📰 Fake News Detection using Machine Learning
=========================================

This project uses machine learning to classify news articles as REAL or FAKE based on their text content. It loads and processes labeled news data from two CSV files, vectorizes the text using TF-IDF, and trains multiple models to choose the most accurate classifier. The final model and vectorizer are saved using joblib for future predictions.

-----------------------------------------
📂 Project Structure
-----------------------------------------

fake-news-detector/
├── Fake.csv                  <- Dataset containing fake news articles
├── True.csv                  <- Dataset containing real news articles
├── fake_news_detector.py     <- Main Python script: training, evaluation, export
├── svm_model.pkl             <- Saved Support Vector Machine model (joblib)
├── tfidf_vectorizer.pkl      <- Saved TF-IDF vectorizer (joblib)
├── requirements.txt          <- List of required Python packages
├── .gitignore                <- Git ignore rules (.venv, __pycache__, etc.)
└── README.txt                <- Project overview and instructions

-----------------------------------------
🧠 ML Workflow Summary
-----------------------------------------

1. Load and Label Data:
   - Fake.csv → label 0
   - True.csv → label 1

2. Preprocess Text:
   - Lowercase, remove URLs, punctuation, and non-alphabet characters

3. Feature Extraction:
   - Apply TF-IDF Vectorization to transform text to numeric features

4. Train & Evaluate Models:
   - Logistic Regression
   - Naive Bayes
   - Random Forest
   - Support Vector Machine (SVM) ← best performing model

5. Export:
   - Save the trained SVM model as svm_model.pkl
   - Save the TF-IDF vectorizer as tfidf_vectorizer.pkl

-----------------------------------------
▶ How to Run
-----------------------------------------

1. Install dependencies:

   pip install -r requirements.txt

2. Run the training and prediction script:

   python fake_news_detector.py

3. Output:
   - The model and vectorizer will be saved as .pkl files
   - Accuracy and confusion matrix will be printed

-----------------------------------------
📌 Notes
-----------------------------------------

- Virtual environment (.venv/) is ignored from Git using .gitignore
- This project is script-based and runs locally
- No frontend or web app included

-----------------------------------------
👨‍💻 Author
-----------------------------------------

Vinay Rahul  
GitHub: https://github.com/vinayrahul203  
Project Repo: https://github.com/vinayrahul203/Fake-News-Prediction
