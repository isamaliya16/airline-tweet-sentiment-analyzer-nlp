# Airline Tweet Sentiment Analyzer (Streamlit app)

Type your own text about an airline and classify it as **negative, neutral or positive** with three approaches:
a keyword **Heuristic**, NLTK **VADER**, and a **TF-IDF + Logistic Regression** model.

## Pages
| Page | What you can do |
|---|---|
| Analyze | Type a text, pick one approach or **Compare all**, see the cleaned text, probabilities, *why* the result was given (matched words, VADER scores, words that pushed the model) and POS tags |
| Batch | Paste many lines or upload a CSV/TXT, analyze them all, view counts, download the results |
| Dataset | Class balance, sentiment by airline, most frequent words, search and filter the tweets |
| Performance | Accuracy, macro-F1, confusion matrix and per-class scores of all approaches on the held-out test split (computed live, nothing hard-coded) |
| History | Every analysis of the current session, downloadable as CSV |

## Run
```bash
pip install -r requirements.txt
streamlit run app.py          # or: streamlit run streamlit/app.py from the repo root
```
NLTK data (`vader_lexicon`, `averaged_perceptron_tagger_eng`) is downloaded automatically on first start.
The Dataset and Performance pages need `Tweets.csv` in `dataset/` (next to this folder or inside it).

## Project structure
```
app.py            entry point: page setup, navigation, sidebar status
config.py         all settings (paths, limits, thresholds, labels)
nlp_utils.py      clean_text() and the heuristic approach
services.py       model/NLTK loading and the three predictors (one Prediction object each)
data_utils.py     dataset loading, statistics, evaluation
train.py          re-train and save the model:  python train.py
lexicon/          editable word lists (one word per line)
views/            one file per page
tests/            pytest tests:  python -m pytest tests
```

## Notes
- Change limits and thresholds in `config.py`; change the heuristic word lists in `lexicon/*.txt`.
- `sentiment_model.joblib` was trained with scikit-learn 1.5.2 (pinned). If it cannot be loaded, the app re-trains
  the same model from `Tweets.csv` automatically and says so in the sidebar.
- Dataset: [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
