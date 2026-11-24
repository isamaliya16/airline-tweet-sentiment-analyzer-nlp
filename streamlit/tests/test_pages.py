"""Run every page headlessly with Streamlit's AppTest and type our own text."""
from streamlit.testing.v1 import AppTest


def page(module):
    script = f'from views.{module} import render\nrender()'
    return AppTest.from_string(script, default_timeout=120).run()


def test_analyze_empty_input_warns():
    at = page('analyze')
    at.button[0].click().run()
    assert at.warning and 'enter some text' in at.warning[0].value


def test_analyze_ml_with_own_text():
    at = page('analyze')
    at.text_area[0].set_value('The pilot was kind but my baggage was damaged and nobody apologised')
    at.button[0].click().run()          # form values are sent together with the click
    assert not at.exception
    assert any('Pushes towards' in m.value for m in at.markdown)
    assert at.session_state['history'][0]['approach'] == 'ML model'


def test_analyze_compare_all_logs_three_results():
    at = page('analyze')
    at.text_area[0].set_value('terrible delay, I want a refund')
    at.radio[0].set_value('Compare all')
    at.button[0].click().run()
    assert not at.exception
    assert len(at.session_state['history']) == 3


def test_batch_typed_lines():
    at = page('batch')
    at.text_area[0].set_value('lovely smooth flight\nmy luggage is gone\n@x !!!').run()
    at.button[0].click().run()
    assert not at.exception
    result, _ = at.session_state['batch_result']
    assert len(result) == 3
    assert result.loc[2, 'ML model'] == '(no text)'


def test_batch_without_text_warns():
    at = page('batch')
    at.button[0].click().run()
    assert at.warning


def test_dataset_page_and_filters():
    at = page('explore')
    assert not at.exception
    at.text_input[0].set_value('refund').run()
    assert any('match' in c.value for c in at.caption)


def test_performance_page_matches_notebook_scores():
    at = page('performance')
    assert not at.exception
    assert any('unseen tweets' in m.value for m in at.markdown)


def test_history_empty_message():
    at = page('history')
    assert at.info
