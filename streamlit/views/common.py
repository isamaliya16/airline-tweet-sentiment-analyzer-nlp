"""Small UI helpers shared by the pages."""
from datetime import datetime

import pandas as pd
import streamlit as st

from config import LABEL_ICONS


def show_label(label, confidence=None):
    """Coloured result box (green / red / yellow)."""
    message = f'{LABEL_ICONS.get(label, "")} {label.capitalize()}'
    if confidence is not None:
        message += f' ({confidence:.0%} confident)'
    if label == 'positive':
        st.success(message)
    elif label == 'negative':
        st.error(message)
    else:
        st.warning(message)


def add_history(text, predictions):
    """Remember every successful prediction of this browser session."""
    history = st.session_state.setdefault('history', [])
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for p in predictions:
        if p.label is not None:
            history.append({'time': now, 'text': text, 'approach': p.approach,
                            'sentiment': p.label, 'confidence': p.confidence})


def need_dataset(df):
    """Show a friendly message and return False when the dataset is missing."""
    if df is None:
        st.warning('`Tweets.csv` was not found, so this page cannot be shown. '
                   'Put it in a `dataset` folder (see README) and reload.')
        return False
    return True


def to_csv(df):
    return pd.DataFrame(df).to_csv(index=False).encode('utf-8')
