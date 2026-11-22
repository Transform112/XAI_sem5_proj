

# T41 core metadata fix: checkpoint period provenance for cross-checkpoint S7 runs
def test_checkpoint_period_comes_from_checkpoint_payload():
    from scripts.run_explain import checkpoint_period_from_payload

    assert checkpoint_period_from_payload({"period": 3}, fallback=1) == 3
    assert checkpoint_period_from_payload({"period": "4"}, fallback=1) == 4
    assert checkpoint_period_from_payload({}, fallback=2) == 2
    assert checkpoint_period_from_payload({"period": None}, fallback=5) == 5
    assert checkpoint_period_from_payload({"period": "bad"}, fallback=7) == 7
