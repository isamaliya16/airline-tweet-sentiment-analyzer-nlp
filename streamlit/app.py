"""Airline Tweet Sentiment Analyzer - entry point.  Run:  streamlit run app.py"""
import streamlit as st

from config import DATASET_URL
from services import system_status
from views import analyze, batch, explore, history, performance

st.set_page_config(page_title='Airline Tweet Sentiment Analyzer', page_icon='✈️', layout='wide')

pages = [
    st.Page(analyze.render, title='Analyze', icon=':material/edit_note:', url_path='analyze', default=True),
    st.Page(batch.render, title='Batch', icon=':material/table_rows:', url_path='batch'),
    st.Page(explore.render, title='Dataset', icon=':material/insights:', url_path='dataset'),
    st.Page(performance.render, title='Performance', icon=':material/speed:', url_path='performance'),
    st.Page(history.render, title='History', icon=':material/history:', url_path='history'),
]
navigation = st.navigation(pages)

with st.sidebar:
    st.title('✈️ Airline Tweet Sentiment')
    st.caption('Classify tweets as negative, neutral or positive with three approaches.')
    st.markdown(f'[Dataset: Twitter US Airline Sentiment]({DATASET_URL})')
    with st.expander('System status'):
        status = system_status()
        for name, ok in status.items():
            if name != 'model_source':
                st.write(('✅ ' if ok else '⚠️ ') + name)
        if status['model_source'] and status['model_source'] != 'saved file':
            st.caption(f'Model was {status["model_source"]}.')

navigation.run()
