from flow_matching.statistics import paired_bootstrap_ci


def test_zero_deltas_have_zero_interval():
    result = paired_bootstrap_ci([0.0, 0.0, 0.0], samples=100)
    assert result["mean"] == 0.0
    assert result["ci_low"] == 0.0
    assert result["ci_high"] == 0.0
