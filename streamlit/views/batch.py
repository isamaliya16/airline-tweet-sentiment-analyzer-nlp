"""Page 2: analyse many texts at once (typed lines or an uploaded file)."""
import pandas as pd
import streamlit as st

from config import APPROACHES, LABELS, MAX_BATCH_ROWS, ML, VADER
from nlp_utils import clean_text
from services import ApproachUnavailable, predict_batch, prepare_text
from views.common import to_csv

NO_TEXT = '(no text)'
TEXT_COLUMN_HINTS = ('text', 'tweet', 'review', 'comment', 'message')


def read_texts():
    """Get the list of texts from the typed box or the uploaded file."""
    source = st.radio('Input', ['Type or paste lines', 'Upload a file'], horizontal=True)
    if source == 'Type or paste lines':
        typed = st.text_area('One text per line', height=200, placeholder='Write each tweet on its own line...')
        return [line.strip() for line in typed.splitlines() if line.strip()]

    uploaded = st.file_uploader('CSV or TXT file', type=['csv', 'txt'])
    if uploaded is None:
        return []
    if uploaded.name.lower().endswith('.txt'):
        content = uploaded.getvalue().decode('utf-8', errors='replace')
        return [line.strip() for line in content.splitlines() if line.strip()]
    try:
        table = pd.read_csv(uploaded)
    except Exception:
        uploaded.seek(0)
        try:
            table = pd.read_csv(uploaded, encoding='latin-1')
        except Exception:
            st.error('This file could not be read as a CSV.')
            return []
    if table.empty:
        st.error('The file has no rows.')
        return []
    names = [str(c) for c in table.columns]
    guess = next((i for i, n in enumerate(names) if n.lower() in TEXT_COLUMN_HINTS), 0)
    column = st.selectbox('Column that holds the text', table.columns, index=guess)
    return table[column].dropna().astype(str).str.strip().loc[lambda s: s != ''].tolist()


def run_analysis(texts, chosen):
    clean = [clean_text(t) for t in texts]
    result = pd.DataFrame({'text': texts, 'cleaned_text': clean})
    empty = result['cleaned_text'] == ''
    for approach in chosen:
        predicted = predict_batch(approach, texts)
        labels = predicted['label'].astype(object)
        if approach != VADER:               # these two work on the cleaned text
            labels[empty] = NO_TEXT
        result[approach] = labels
        if approach == ML:
            result['ML confidence'] = predicted['confidence'].round(3)
    return result


def render():
    st.header('Batch analysis')
    st.write('Analyze many texts together and download the results.')

    texts = read_texts()
    chosen = st.multiselect('Approaches', APPROACHES, default=APPROACHES)

    if len(texts) > MAX_BATCH_ROWS:
        st.info(f'Only the first {MAX_BATCH_ROWS} of {len(texts)} texts will be analyzed.')
        texts = texts[:MAX_BATCH_ROWS]
    if texts:
        st.caption(f'{len(texts)} text(s) ready.')

    if st.button('Run analysis', type='primary'):
        if not texts:
            st.warning('Please enter or upload some text first.')
        elif not chosen:
            st.warning('Please select at least one approach.')
        else:
            texts = [prepare_text(t)[0] for t in texts]
            try:
                with st.spinner('Analyzing...'):
                    st.session_state['batch_result'] = (run_analysis(texts, chosen), chosen)
            except ApproachUnavailable as problem:
                st.error(str(problem))

    if 'batch_result' in st.session_state:
        show_batch_result(*st.session_state['batch_result'])


def show_batch_result(result, chosen):
    st.divider()
    st.subheader('Results')
    counts = pd.DataFrame({a: result[a].value_counts() for a in chosen}).reindex(LABELS).fillna(0).astype(int)
    st.write('**Number of texts per sentiment**')
    st.bar_chart(counts)

    if len(chosen) > 1:
        same = (result[chosen].nunique(axis=1) == 1).mean()
        st.caption(f'All selected approaches give the same label for {same:.0%} of the texts.')

    st.dataframe(result, hide_index=True)
    st.download_button('Download results (CSV)', to_csv(result), 'sentiment_results.csv', 'text/csv')
