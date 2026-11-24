"""Resource loading and the three prediction approaches.

Every approach returns the same `Prediction` object, so the pages can show
them in the same way.
"""
from dataclasses import dataclass, field

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

from config import (APPROACHES, HEURISTIC, LABELS, MAX_CHARS, ML, MODEL_PATH, NLTK_RESOURCES,
                    POS_DESCRIPTIONS, TOP_FEATURES, VADER, VADER_THRESHOLD)
from data_utils import find_dataset_path, load_dataset
from nlp_utils import clean_text, heuristic_matches, heuristic_sentiment


class ApproachUnavailable(RuntimeError):
    """Raised when an approach cannot run (missing model or NLTK data)."""


@dataclass
class Prediction:
    approach: str
    label: str | None = None
    confidence: float | None = None          # probability, only for the ML model
    scores: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)
    error: str | None = None


# ---- Cached resources ------------------------------------------------------
@st.cache_resource
def ensure_nltk_data():
    """Download missing NLTK data once and return the resources that are available."""
    available = set()
    for resource, path in NLTK_RESOURCES.items():
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(resource, quiet=True)
            except Exception:
                pass
        try:
            nltk.data.find(path)
            available.add(resource)
        except LookupError:
            pass
    return frozenset(available)


@st.cache_data(show_spinner='Loading dataset...')
def get_dataset():
    return load_dataset()


@st.cache_resource(show_spinner='Loading model...')
def load_model():
    """Return (model, source). If the saved file cannot be loaded (for example a
    scikit-learn version mismatch) the model is re-trained from the dataset."""
    try:
        return joblib.load(MODEL_PATH), 'saved file'
    except Exception:
        pass
    data = load_dataset()
    if data is None:
        return None, None
    from train import train_model
    return train_model(data), 're-trained from dataset'


@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()


def system_status():
    """What is available right now (shown in the sidebar)."""
    nltk_ready = ensure_nltk_data()
    model, model_source = load_model()
    return {
        'VADER lexicon': 'vader_lexicon' in nltk_ready,
        'POS tagger': 'averaged_perceptron_tagger_eng' in nltk_ready,
        'ML model': model is not None,
        'Dataset': find_dataset_path() is not None,
        'model_source': model_source,
    }


# ---- Helpers ---------------------------------------------------------------
def prepare_text(text):
    """Strip the text and cut it to MAX_CHARS. Returns (text, was_truncated)."""
    text = (text or '').strip()
    return text[:MAX_CHARS], len(text) > MAX_CHARS


def vader_label(compound):
    if compound >= VADER_THRESHOLD:
        return 'positive'
    if compound <= -VADER_THRESHOLD:
        return 'negative'
    return 'neutral'


def _vader():
    if 'vader_lexicon' not in ensure_nltk_data():
        raise ApproachUnavailable('The VADER lexicon could not be downloaded. Check the internet connection.')
    return load_vader()


def _model():
    model, _ = load_model()
    if model is None:
        raise ApproachUnavailable('The ML model could not be loaded.')
    return model


# ---- Single text -----------------------------------------------------------
def explain_ml(model, cleaned, label):
    """Words (and word pairs) that pushed the model towards / away from `label`."""
    vectorizer, classifier = model.steps[0][1], model.steps[-1][1]
    row = vectorizer.transform([cleaned])
    class_index = list(classifier.classes_).index(label)
    names = vectorizer.get_feature_names_out()[row.indices]
    table = pd.DataFrame({'word': names, 'effect': row.data * classifier.coef_[class_index][row.indices]})
    table = table.sort_values('effect', ascending=False)
    return {
        'known_terms': int(row.nnz),
        'supporting': table[table['effect'] > 0].head(TOP_FEATURES).reset_index(drop=True),
        'opposing': table[table['effect'] < 0].tail(TOP_FEATURES).iloc[::-1].reset_index(drop=True),
    }


def predict_heuristic(text):
    matches = heuristic_matches(text)
    return Prediction(HEURISTIC, heuristic_sentiment(text),
                      details={'matches': matches, 'score': sum(m['value'] for m in matches)})


def predict_vader(text):
    scores = _vader().polarity_scores(text)
    return Prediction(VADER, vader_label(scores['compound']), scores=scores)


def predict_ml(text):
    model = _model()
    cleaned = clean_text(text)
    if not cleaned:
        return Prediction(ML, error='Nothing is left after cleaning (only symbols, links or mentions).')
    probabilities = dict(zip(model.classes_, model.predict_proba([cleaned])[0]))
    label = max(probabilities, key=probabilities.get)
    return Prediction(ML, label, confidence=probabilities[label], scores=probabilities,
                      details=explain_ml(model, cleaned, label))


PREDICTORS = {HEURISTIC: predict_heuristic, VADER: predict_vader, ML: predict_ml}


def analyze(approach, text):
    """Run one approach on one text. Problems are returned in `Prediction.error`."""
    try:
        return PREDICTORS[approach](text)
    except ApproachUnavailable as problem:
        return Prediction(approach, error=str(problem))


def pos_tags(cleaned):
    """POS table for the cleaned text, or None if the tagger data is missing."""
    if 'averaged_perceptron_tagger_eng' not in ensure_nltk_data() or not cleaned:
        return None
    tagged = nltk.pos_tag(cleaned.split())
    return pd.DataFrame([(w, t, POS_DESCRIPTIONS.get(t, '')) for w, t in tagged],
                        columns=['Word', 'Tag', 'Meaning'])


# ---- Many texts ------------------------------------------------------------
def predict_batch(approach, texts):
    """Labels (and ML confidence) for a list of raw texts as a DataFrame."""
    confidence = [None] * len(texts)
    if approach == HEURISTIC:
        labels = [heuristic_sentiment(t) for t in texts]
    elif approach == VADER:
        analyzer = _vader()
        labels = [vader_label(analyzer.polarity_scores(t)['compound']) for t in texts]
    elif approach == ML:
        model = _model()
        probabilities = model.predict_proba([clean_text(t) for t in texts])
        labels = [model.classes_[i] for i in probabilities.argmax(axis=1)]
        confidence = probabilities.max(axis=1)
    else:
        raise ValueError(f'Unknown approach: {approach}')
    return pd.DataFrame({'label': labels, 'confidence': confidence})


def predict_labels(approach, texts):
    return predict_batch(approach, texts)['label'].tolist()
