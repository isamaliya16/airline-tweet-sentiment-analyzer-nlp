"""Dataset loading, simple statistics and evaluation of the three approaches."""
from collections import Counter

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

from config import APPROACHES, DATASET_CANDIDATES, LABELS, RANDOM_STATE, TEST_SIZE
from nlp_utils import clean_text, negation_words

COLUMNS = ['text', 'airline_sentiment', 'airline']


def find_dataset_path():
    for path in DATASET_CANDIDATES:
        if path.exists():
            return path
    return None


def load_dataset():
    """Load Tweets.csv exactly like the notebook. Returns None if the file is missing."""
    path = find_dataset_path()
    if path is None:
        return None
    df = pd.read_csv(path)[COLUMNS].drop_duplicates().reset_index(drop=True)
    df['clean_text'] = df['text'].apply(clean_text)
    df = df[df['clean_text'] != ''].reset_index(drop=True)
    df['word_count'] = df['text'].str.split().str.len()
    return df


def split_dataset(df):
    """80/20 stratified split used for training and for every evaluation."""
    return train_test_split(
        df[['text', 'clean_text']], df['airline_sentiment'],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['airline_sentiment'])


def top_words(df, sentiment, n=15):
    """Most frequent words of one sentiment (stop-words removed, negations kept)."""
    stop = set(ENGLISH_STOP_WORDS) - negation_words
    counter = Counter()
    for text in df.loc[df['airline_sentiment'] == sentiment, 'clean_text']:
        counter.update(w for w in text.split() if w not in stop and len(w) > 1)
    return pd.DataFrame(counter.most_common(n), columns=['word', 'count'])


def evaluate_approaches(predict_fn, df):
    """Score every approach on the held-out test split.

    predict_fn(approach, texts) -> list of labels (raw tweets in, labels out).
    Returns {approach: {'accuracy', 'macro_f1', 'report', 'matrix'}}.
    """
    _, x_test, _, y_test = split_dataset(df)
    results = {}
    for approach in APPROACHES:
        try:
            pred = predict_fn(approach, x_test['text'].tolist())
        except RuntimeError:          # approach not available (e.g. missing NLTK data)
            continue
        results[approach] = {
            'accuracy': accuracy_score(y_test, pred),
            'macro_f1': f1_score(y_test, pred, average='macro'),
            'report': pd.DataFrame(classification_report(
                y_test, pred, labels=LABELS, output_dict=True, zero_division=0)).T,
            'matrix': pd.DataFrame(
                confusion_matrix(y_test, pred, labels=LABELS),
                index=[f'true {l}' for l in LABELS], columns=[f'pred {l}' for l in LABELS]),
        }
    results['test_size'] = len(y_test)
    return results
