import joblib
import nltk
import pandas as pd
import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer

from nlp_utils import clean_text, heuristic_sentiment

st.set_page_config(page_title='Airline Tweet Sentiment Analyzer')

# NLTK data is downloaded quietly at start-up
for resource in ['vader_lexicon', 'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

MAX_CHARS = 500
examples = [
    'Thanks to the crew, the flight was smooth and on time!',
    'Flight delayed again and my bag is lost. Not happy with the service.',
    'What is the baggage allowance for international flights?',
]


@st.cache_resource
def load_model():
    return joblib.load('sentiment_model.joblib')


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


st.title('Airline Tweet Sentiment Analyzer')
st.sidebar.header('About')
st.sidebar.write('NLP PR 1 - Red & White Skill Education. Sentiment of airline tweets using three approaches.')
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
            show_label(vader_label(tweet))
        elif not cleaned:
            st.warning('Nothing is left after cleaning (only symbols, links or mentions). Try a longer sentence.')
        elif approach == 'Heuristic':
            show_label(heuristic_sentiment(tweet))
        else:
            model = load_model()
            show_label(model.predict([cleaned])[0])
            probabilities = pd.Series(model.predict_proba([cleaned])[0], index=model.classes_)
            st.bar_chart(probabilities)

        if cleaned:
            with st.expander('POS tags'):
                tags = nltk.pos_tag(cleaned.split())
                st.dataframe(pd.DataFrame(tags, columns=['Word', 'Tag']))
