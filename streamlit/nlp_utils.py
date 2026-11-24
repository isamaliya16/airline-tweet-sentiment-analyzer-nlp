"""Text cleaning and the rule-based (heuristic) sentiment approach.

Word lists live in the `lexicon/` folder (one word per line) so they can be
edited without touching the code.
"""
import re

from config import LEXICON_DIR, NEGATION_WINDOW


def load_words(file_name):
    """Read a word list file into a set (blank lines and # comments ignored)."""
    path = LEXICON_DIR / file_name
    lines = path.read_text(encoding='utf-8').splitlines()
    return {line.strip() for line in lines if line.strip() and not line.startswith('#')}


positive_words = load_words('positive_words.txt')
negative_words = load_words('negative_words.txt')
negation_words = load_words('negation_words.txt')


def clean_text(s):
    s = s.lower()
    s = re.sub(r'https?://\S+', ' ', s)      # remove URLs
    s = re.sub(r'@\w+', ' ', s)              # remove @mentions
    s = re.sub(r'#(\w+)', r'\1', s)          # keep hashtag word
    s = re.sub(r"['’‘]", '', s)             # don't -> dont
    s = re.sub(r'[^a-z\s]', ' ', s)          # numbers, punctuation, emojis
    s = re.sub(r'\s+', ' ', s).strip()       # extra spaces
    return s


def heuristic_matches(text):
    """Return every lexicon word found in the text with its effect on the score.

    Each item: {'word', 'value' (+1/-1 after negation), 'negated' (bool)}.
    """
    words = clean_text(text).split()
    matches = []
    for i, w in enumerate(words):
        if w in positive_words:
            value = 1
        elif w in negative_words:
            value = -1
        else:
            continue
        negated = any(p in negation_words for p in words[max(0, i - NEGATION_WINDOW):i])
        matches.append({'word': w, 'value': -value if negated else value, 'negated': negated})
    return matches


def heuristic_score(text):
    return sum(m['value'] for m in heuristic_matches(text))


def heuristic_sentiment(text):
    score = heuristic_score(text)
    if score > 0:
        return 'positive'
    if score < 0:
        return 'negative'
    return 'neutral'
