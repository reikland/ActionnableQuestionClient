from pathlib import Path


def test_prompt_contracts_present():
    txt = Path('app.py').read_text(encoding='utf-8')
    assert 'STRATEGIC AXES:' in txt
    assert 'QUESTIONS:' in txt
    assert 'Metaculus-style' in txt


def test_online_toggle_exists():
    txt = Path('app.py').read_text(encoding='utf-8')
    assert 'use_online_research' in txt
    assert ':online' in txt
