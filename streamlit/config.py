"""Central settings. Change values here instead of editing the app code."""
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ---- Files ---------------------------------------------------------------
MODEL_PATH = BASE_DIR / 'sentiment_model.joblib'
LEXICON_DIR = BASE_DIR / 'lexicon'
# The first existing file is used (works locally and on Streamlit Cloud).
DATASET_CANDIDATES = [
    BASE_DIR / 'dataset' / 'Tweets.csv',
    BASE_DIR.parent / 'dataset' / 'Tweets.csv',
]
DATASET_URL = 'https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment'

# ---- NLTK data: resource name -> path used by nltk.data.find() -------------
NLTK_RESOURCES = {
    'vader_lexicon': 'sentiment/vader_lexicon.zip/vader_lexicon/vader_lexicon.txt',
    'averaged_perceptron_tagger_eng': 'taggers/averaged_perceptron_tagger_eng/',
}

# ---- Approaches and labels -------------------------------------------------
HEURISTIC, VADER, ML = 'Heuristic', 'VADER', 'ML model'
APPROACHES = [HEURISTIC, VADER, ML]
COMPARE_ALL = 'Compare all'
LABELS = ['negative', 'neutral', 'positive']
LABEL_ICONS = {'negative': '🔴', 'neutral': '🟡', 'positive': '🟢'}

# ---- Input limits ----------------------------------------------------------
MAX_CHARS = 500          # single tweet
MAX_BATCH_ROWS = 2000    # rows processed in the batch page

# ---- Model behaviour -------------------------------------------------------
VADER_THRESHOLD = 0.05        # |compound| below this -> neutral
NEGATION_WINDOW = 2           # words before a sentiment word checked for negation
LOW_CONFIDENCE = 0.5          # below this the ML result is flagged as uncertain
TOP_FEATURES = 8              # words shown in the "why" table

# ---- Evaluation (same split as the notebook) -------------------------------
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ---- Short descriptions of common Penn Treebank POS tags -------------------
POS_DESCRIPTIONS = {
    'NN': 'noun', 'NNS': 'noun (plural)', 'NNP': 'proper noun', 'VB': 'verb',
    'VBD': 'verb (past)', 'VBG': 'verb (-ing)', 'VBN': 'verb (past participle)',
    'VBP': 'verb (present)', 'VBZ': 'verb (3rd person)', 'JJ': 'adjective',
    'JJR': 'adjective (comparative)', 'JJS': 'adjective (superlative)',
    'RB': 'adverb', 'PRP': 'pronoun', 'PRP$': 'possessive pronoun',
    'DT': 'determiner', 'IN': 'preposition', 'CC': 'conjunction',
    'MD': 'modal verb', 'TO': '"to"', 'CD': 'number', 'WRB': 'wh-adverb',
    'WP': 'wh-pronoun', 'UH': 'interjection',
}
