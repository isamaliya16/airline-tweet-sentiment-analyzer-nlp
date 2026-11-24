"""Train (or re-train) the TF-IDF + Logistic Regression model.

Usage:  python train.py
It uses the same cleaning, split and settings as the notebook, prints the test
scores and saves the pipeline to MODEL_PATH.
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from config import MODEL_PATH
from data_utils import load_dataset, split_dataset


def build_pipeline():
    return Pipeline([
        ('vec', TfidfVectorizer(ngram_range=(1, 2), min_df=2)),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced')),
    ])


def train_model(df):
    x_train, _, y_train, _ = split_dataset(df)
    return build_pipeline().fit(x_train['clean_text'], y_train)


if __name__ == '__main__':
    data = load_dataset()
    if data is None:
        raise SystemExit('Tweets.csv not found. See DATASET_CANDIDATES in config.py.')
    model = train_model(data)
    _, x_test, _, y_test = split_dataset(data)
    pred = model.predict(x_test['clean_text'])
    print(f'Accuracy {accuracy_score(y_test, pred):.3f} | macro-F1 {f1_score(y_test, pred, average="macro"):.3f}')
    joblib.dump(model, MODEL_PATH)
    print('Saved model to', MODEL_PATH)
