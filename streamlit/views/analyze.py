"""Page 1: analyse one text typed by the user."""
import pandas as pd
import streamlit as st

from config import (APPROACHES, COMPARE_ALL, HEURISTIC, LABELS, LOW_CONFIDENCE, MAX_CHARS, ML, VADER,
                    VADER_THRESHOLD)
from nlp_utils import clean_text
from services import analyze, pos_tags, prepare_text
from views.common import add_history, show_label


def render():
    st.header('Analyze a tweet')
    st.write('Type or paste any text about an airline, choose an approach and press **Analyze**.')

    previous = st.session_state.get('analysis', {})
    with st.form('analyze_form'):
        text = st.text_area('Your text', value=previous.get('raw', ''), height=140, max_chars=MAX_CHARS,
                            placeholder='For example: write a tweet in your own words...')
        approach = st.radio('Approach', APPROACHES + [COMPARE_ALL], horizontal=True,
                            index=(APPROACHES + [COMPARE_ALL]).index(previous.get('approach', ML)))
        submitted = st.form_submit_button('Analyze', type='primary')

    if submitted:
        text, truncated = prepare_text(text)
        if not text:
            st.warning('Please enter some text.')
        else:
            if truncated:
                st.info(f'The text was cut to {MAX_CHARS} characters.')
            chosen = APPROACHES if approach == COMPARE_ALL else [approach]
            predictions = [analyze(a, text) for a in chosen]
            add_history(text, predictions)
            st.session_state['analysis'] = {'raw': text, 'approach': approach, 'predictions': predictions}
            previous = st.session_state['analysis']

    if previous:
        show_results(previous['raw'], previous['predictions'])


def show_results(text, predictions):
    cleaned = clean_text(text)
    st.divider()
    st.markdown('**Cleaned text:**')
    st.code(cleaned or '(empty)', language=None, wrap_lines=True)

    columns = st.columns(len(predictions))
    for column, prediction in zip(columns, predictions):
        with column:
            st.subheader(prediction.approach)
            if prediction.error:
                st.error(prediction.error)
            else:
                show_label(prediction.label, prediction.confidence)

    labels = {p.approach: p.label for p in predictions if p.label}
    if len(predictions) > 1 and labels:
        if len(set(labels.values())) == 1 and len(labels) == len(predictions):
            st.success(f'All approaches agree: **{next(iter(labels.values()))}**.')
        else:
            summary = ', '.join(f'{name}: {label}' for name, label in labels.items())
            st.info(f'The approaches disagree ({summary}). Short or mixed texts are hard for every method.')

    st.subheader('Why this result?')
    ready = [p for p in predictions if not p.error]
    if len(ready) == 1:
        show_details(ready[0])
    elif ready:
        for tab, prediction in zip(st.tabs([p.approach for p in ready]), ready):
            with tab:
                show_details(prediction)

    with st.expander('Part-of-speech tags'):
        tags = pos_tags(cleaned)
        if tags is None:
            st.info('POS tags are not available (no text or the tagger data is missing).')
        else:
            st.dataframe(tags, hide_index=True)


def show_details(prediction):
    if prediction.approach == HEURISTIC:
        details_heuristic(prediction)
    elif prediction.approach == VADER:
        details_vader(prediction)
    else:
        details_ml(prediction)


def details_heuristic(prediction):
    matches = prediction.details['matches']
    st.write(f'Score = positive words minus negative words = **{prediction.details["score"]:+d}**.')
    if not matches:
        st.info('None of the words in the sentiment lists were found, so the result is neutral.')
        return
    table = pd.DataFrame({
        'Word': [m['word'] for m in matches],
        'Effect': ['positive (+1)' if m['value'] > 0 else 'negative (-1)' for m in matches],
        'Flipped by a negation word': ['yes' if m['negated'] else 'no' for m in matches],
    })
    st.dataframe(table, hide_index=True)


def details_vader(prediction):
    scores = prediction.scores
    columns = st.columns(4)
    columns[0].metric('Negative share', f'{scores["neg"]:.0%}')
    columns[1].metric('Neutral share', f'{scores["neu"]:.0%}')
    columns[2].metric('Positive share', f'{scores["pos"]:.0%}')
    columns[3].metric('Compound score', f'{scores["compound"]:+.2f}')
    st.caption(f'Compound ≥ {VADER_THRESHOLD} is positive, ≤ -{VADER_THRESHOLD} is negative, otherwise neutral.')


def details_ml(prediction):
    details = prediction.details
    st.write('**Class probabilities**')
    st.bar_chart(pd.Series(prediction.scores).reindex(LABELS))
    if details['known_terms'] == 0:
        st.warning('The model does not know any of these words, so this result is only a guess.')
    elif prediction.confidence < LOW_CONFIDENCE:
        st.warning('The model is not very sure about this text. Treat the result with care.')

    left, right = st.columns(2)
    with left:
        st.write(f'**Pushes towards "{prediction.label}"**')
        st.dataframe(details['supporting'], hide_index=True)
    with right:
        st.write(f'**Pushes away from "{prediction.label}"**')
        if details['opposing'].empty:
            st.caption('Nothing in this text points the other way.')
        else:
            st.dataframe(details['opposing'], hide_index=True)
    st.caption('Higher numbers mean a stronger push. Two-word pairs such as "my bag" are also used by the model.')
