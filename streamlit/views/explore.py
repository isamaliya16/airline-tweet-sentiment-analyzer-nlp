"""Page 3: look at the dataset behind the model."""
import streamlit as st

from config import DATASET_URL, LABELS
from data_utils import top_words
from services import get_dataset
from views.common import need_dataset


def render():
    st.header('Explore the dataset')
    df = get_dataset()
    if not need_dataset(df):
        return
    st.caption(f'Source: [Twitter US Airline Sentiment]({DATASET_URL}). Duplicate tweets are removed.')

    share = df['airline_sentiment'].value_counts(normalize=True)
    columns = st.columns(4)
    columns[0].metric('Tweets', f'{len(df):,}')
    columns[1].metric('Airlines', df['airline'].nunique())
    columns[2].metric('Average words per tweet', f'{df["word_count"].mean():.1f}')
    columns[3].metric('Most common sentiment', f'{share.idxmax()} ({share.max():.0%})')

    left, right = st.columns(2)
    with left:
        st.write('**Tweets per sentiment**')
        st.bar_chart(df['airline_sentiment'].value_counts().reindex(LABELS))
    with right:
        st.write('**Sentiment by airline**')
        st.bar_chart(df.groupby(['airline', 'airline_sentiment']).size().unstack(fill_value=0)[LABELS])

    st.subheader('Most frequent words')
    sentiment, amount = st.columns(2)
    chosen = sentiment.selectbox('Sentiment', LABELS, index=0)
    n = amount.slider('Number of words', 5, 30, 15)
    st.bar_chart(top_words(df, chosen, n).set_index('word'))

    st.subheader('Browse tweets')
    airlines = st.multiselect('Airline', sorted(df['airline'].unique()))
    sentiments = st.multiselect('Sentiment', LABELS)
    keyword = st.text_input('Contains the word or phrase')
    view = df
    if airlines:
        view = view[view['airline'].isin(airlines)]
    if sentiments:
        view = view[view['airline_sentiment'].isin(sentiments)]
    if keyword.strip():
        view = view[view['text'].str.contains(keyword.strip(), case=False, regex=False)]
    st.caption(f'{len(view):,} tweet(s) match.')
    st.dataframe(view[['airline', 'airline_sentiment', 'text']], hide_index=True, height=350)
