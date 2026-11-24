"""Page 5: everything analyzed in this browser session."""
import pandas as pd
import streamlit as st

from views.common import to_csv


def render():
    st.header('History')
    history = st.session_state.get('history', [])
    if not history:
        st.info('Nothing here yet. Analyze a text on the first page and it will appear here.')
        return
    table = pd.DataFrame(history)
    table['confidence'] = table['confidence'].round(3)
    st.caption(f'{len(table)} result(s) in this session. They are not saved on the server.')
    st.dataframe(table.iloc[::-1], hide_index=True)
    download, clear = st.columns([1, 5])
    download.download_button('Download CSV', to_csv(table), 'history.csv', 'text/csv')
    if clear.button('Clear history'):
        st.session_state['history'] = []
        st.rerun()
