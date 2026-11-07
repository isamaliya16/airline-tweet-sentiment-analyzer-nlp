from pathlib import Path

import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

from nlp_utils import clean_text, heuristic_sentiment

st.set_page_config(page_title='Airline Tweet Sentiment Analyzer')

# Resolve files relative to this script, not the current working directory,
# so the app also works when started from the repo root (e.g. Streamlit Cloud).
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / 'sentiment_model.joblib'

# NLTK resource -> path used by nltk.data.find() to check if it is installed
NLTK_RESOURCES = {
    'vader_lexicon': 'sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt',
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng/',
}

MAX_CHARS = 500
examples = [
    'Thanks to the crew, the flight was smooth and on time!',
    'Flight delayed again and my bag is lost. Not happy with the service.',
    'What is the baggage allowance for international flights?',
]


@st.cache_resource
def ensure_nltk_data():
    """Download missing NLTK data once per server process.

    Returns the set of resources that are available. nltk.download() returns
    False (instead of raising) on failure, so availability is re-checked.
    """
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


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_vader():
    return SentimentIntensityAnalyzer()


def vader_label(text):
    compound = load_vader().polarity_scores(text)['compound']
    if compound >= 0.05:
        return 'positive'
    if compound <= -0.05:
        return 'negative'
    return 'neutral'


def show_label(label):
    message = f'Predicted sentiment: {label}'
    if label == 'positive':
        st.success(message)
    elif label == 'negative':
        st.error(message)
    else:
        st.warning(message)


def set_example(text):
    st.session_state['tweet'] = text


nltk_ready = ensure_nltk_data()

st.title('Airline Tweet Sentiment Analyzer')
st.sidebar.header('About')
st.sidebar.markdown('[Dataset: Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)')

st.text_area('Enter a tweet', key='tweet', height=120)

columns = st.columns(len(examples))
for i, (col, text) in enumerate(zip(columns, examples)):
    col.button(f'Example {i + 1}', key=f'ex{i}', on_click=set_example, args=(text,))

approach = st.radio('Approach', ['Heuristic', 'VADER', 'ML model'], horizontal=True)

if st.button('Predict'):
    tweet = st.session_state.get('tweet', '').strip()

    if not tweet:
        st.warning('Please enter some text.')
    else:
        if len(tweet) > MAX_CHARS:
            tweet = tweet[:MAX_CHARS]
            st.info(f'Input was longer than {MAX_CHARS} characters and was truncated.')

        cleaned = clean_text(tweet)
        st.write('**Cleaned text:**', cleaned if cleaned else '(empty)')

        if approach == 'VADER':
            if 'vader_lexicon' in nltk_ready:
                show_label(vader_label(tweet))
            else:
                st.error('The VADER lexicon could not be downloaded. Check the internet connection and reload.')
        elif not cleaned:
            st.warning('Nothing is left after cleaning (only symbols, links or mentions). Try a longer sentence.')
        elif approach == 'Heuristic':
            show_label(heuristic_sentiment(tweet))
        else:
            model = load_model()
            probabilities = pd.Series(model.predict_proba([cleaned])[0], index=model.classes_)
            show_label(probabilities.idxmax())
            st.bar_chart(probabilities)

        if cleaned:
            with st.expander('POS tags'):
                if 'averaged_perceptron_tagger_eng' in nltk_ready:
                    tags = nltk.pos_tag(cleaned.split())
                    st.dataframe(pd.DataFrame(tags, columns=['Word', 'Tag']))
                else:
                    st.info('POS tagger data could not be downloaded, so tags are unavailable.')
