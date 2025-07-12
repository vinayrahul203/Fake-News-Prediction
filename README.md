Fake News Detection Project

Overview

This project uses machine learning to detect fake news articles. It trains and evaluates several models on a dataset of labeled news articles and selects the best-performing model for prediction.

Dataset

The dataset consists of two CSV files:

- Fake.csv: contains fake news articles
- True.csv: contains true news articles

Each article is labeled as either 0 (Fake) or 1 (True). The dataset is preprocessed using TF-IDF vectorization.

Models

The project trains and evaluates the following models:

- Logistic Regression
- Naive Bayes
- Random Forest
- Passive Aggressive
- Linear SVC

Requirements

- Python 3.x
- pandas
- scikit-learn
- numpy

Usage

1. Clone the repository:https://github.com/vinayrahul203/Fake-News-Prediction.git
2. Run the code: python fake_news_detection.py

Results

The project outputs the accuracy of each model and selects the best-performing model for prediction. The best model's predictions are then used to label the full dataset.

Code Structure

- fake_news_detection.py: contains the main code for data loading, model training, and evaluation
- README.md: this file

Author

- https://github.com/vinayrahul203

