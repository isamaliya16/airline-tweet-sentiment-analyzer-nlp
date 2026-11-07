# Airline Tweet Sentiment Analyzer

Streamlit app that classifies airline tweets as positive / neutral / negative using
three approaches: a keyword heuristic, NLTK VADER, and a TF-IDF + Logistic Regression model.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py          # or: streamlit run streamlit/app.py from the repo root
```
NLTK data (`vader_lexicon`, `averaged_perceptron_tagger_eng`) is downloaded automatically on first start.

## Notes
- `sentiment_model.joblib` was trained with scikit-learn 1.5.2; keep the pinned version in `requirements.txt`.
- Dataset: [Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)
