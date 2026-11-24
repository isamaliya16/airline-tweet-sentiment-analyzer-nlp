"""Logic tests with hand-written texts (no example data from the app)."""
from nlp_utils import clean_text, heuristic_matches, heuristic_sentiment
from services import analyze, predict_batch, prepare_text


def test_clean_text_removes_noise_and_keeps_negation():
    raw = "@Delta Why didn't you tell me?! Check https://t.co/xyz #lostbag 123"
    assert clean_text(raw) == 'why didnt you tell me check lostbag'


def test_heuristic_negation_flips_meaning():
    assert heuristic_sentiment('I am not happy with this') == 'negative'
    assert heuristic_sentiment('this was not bad') == 'positive'
    assert heuristic_sentiment('where is gate twelve') == 'neutral'


def test_heuristic_matches_report_negation():
    matches = heuristic_matches('never rude, very friendly')
    assert {'word': 'rude', 'value': 1, 'negated': True} in matches
    assert {'word': 'friendly', 'value': 1, 'negated': False} in matches


def test_prepare_text_truncates():
    text, truncated = prepare_text('a' * 900)
    assert len(text) == 500 and truncated
    assert prepare_text('   ') == ('', False)


def test_ml_prediction_has_probabilities_and_explanation():
    result = analyze('ML model', 'my suitcase never arrived and the staff were useless')
    assert result.label == 'negative'
    assert abs(sum(result.scores.values()) - 1) < 1e-6
    assert result.details['known_terms'] > 0


def test_ml_symbol_only_text_gives_error():
    assert analyze('ML model', '@united !!! 😡').error


def test_vader_and_batch_agree_on_one_text():
    text = 'the crew was wonderful and kind'
    single = analyze('VADER', text).label
    assert predict_batch('VADER', [text])['label'][0] == single == 'positive'
