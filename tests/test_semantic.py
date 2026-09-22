from fly_codex_lab.semantic import keyword_encode, random_encode

def test_keyword_threat_signal():
    v = keyword_encode('市場突然暴跌，大家非常恐慌')
    assert v.threat > 0.12
    assert v.arousal > 0.12

def test_random_is_reproducible():
    assert random_encode('hello', 5175) == random_encode('hello', 5175)
