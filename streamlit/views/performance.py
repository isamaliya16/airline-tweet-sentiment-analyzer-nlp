"""Page 4: how good is each approach? (measured live on the held-out test split)"""
import pandas as pd
import streamlit as st

from config import APPROACHES, ML
from data_utils import evaluate_approaches
from services import get_dataset, load_model, predict_labels
from views.common import need_dataset


@st.cache_data(show_spinner='Evaluating all approaches on the test set...')
def run_evaluation():
    return evaluate_approaches(predict_labels, get_dataset())


def render():
    st.header('Model performance')
    if not need_dataset(get_dataset()):
        return
    results = run_evaluation()
    available = [a for a in APPROACHES if a in results]
    if not available:
        st.error('No approach could be evaluated.')
        return
    st.write(f'All approaches are scored on the same **{results["test_size"]:,}** unseen tweets '
             '(20% of the data, never used for training).')

    summary = pd.DataFrame({
        'Approach': available,
        'Accuracy': [results[a]['accuracy'] for a in available],
        'Macro-F1': [results[a]['macro_f1'] for a in available],
    }).set_index('Approach')
    st.dataframe(summary.style.format('{:.3f}'))
    st.bar_chart(summary['Macro-F1'])
    st.caption('Macro-F1 averages the F1 score of the three classes, so the small positive class counts as '
               'much as the large negative class. Accuracy alone would hide that.')

    st.subheader('Details')
    approach = st.selectbox('Approach', available, index=available.index(ML) if ML in available else 0)
    matrix, report = st.columns(2)
    with matrix:
        st.write('**Confusion matrix** (rows = true, columns = predicted)')
        st.dataframe(results[approach]['matrix'])
    with report:
        st.write('**Precision, recall and F1 per class**')
        st.dataframe(results[approach]['report'].round(3))

    model, source = load_model()
    if model is not None:
        with st.expander('About the ML model'):
            vectorizer, classifier = model.steps[0][1], model.steps[-1][1]
            st.write(f'Loaded from: {source}')
            st.write(f'Features: {type(vectorizer).__name__}, {len(vectorizer.vocabulary_):,} terms, '
                     f'n-gram range {vectorizer.ngram_range}, min_df {vectorizer.min_df}')
            st.write(f'Classifier: {type(classifier).__name__} (class_weight = {classifier.class_weight})')
