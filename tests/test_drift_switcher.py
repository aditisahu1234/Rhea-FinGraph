"""Unit tests for the concept-drift auto-switcher (Layer 4)."""

from __future__ import annotations

from fingraph_sentinel.drift_switcher import (
    ADWIN,
    PageHinkley,
    persist_decision,
    rank_candidates,
    run_auto_switch,
)


def test_stable_stream_no_switch():
    stable = [0.5] * 12
    d = run_auto_switch(stable)
    assert d.triggered is False


def test_step_change_triggers_switch():
    jump = [0.5] * 8 + [0.9] * 6
    d = run_auto_switch(jump)
    assert d.triggered
    assert d.from_model == "baseline-online-xgb"
    assert d.to_model in {"baseline-online-v3", "baseline-full-xgb"}
    assert any(a.detector == "page-hinkley" for a in d.alerts)


def test_page_hinkley_fires_on_shift():
    ph = PageHinkley(delta=0.05, lamb=2.0)
    ph.set_reference(0.5)
    fired = [ph.update(0.5) for _ in range(3)] + [ph.update(0.9) for _ in range(6)]
    assert any(fired)


def test_adwin_does_not_fire_on_flat():
    ad = ADWIN(delta=0.05)
    fired = [ad.update(0.5) for _ in range(12)]
    assert not any(fired)


def test_adwin_shrinks_window_on_sustained_shift():
    # ADWIN's Hoeffding cut is conservative by design (its ``delta`` is a
    # confidence, so a small sample needs a large shift to be significant).
    # With a big, sustained jump it must eventually prune the old window.
    ad = ADWIN(delta=0.9)  # low confidence requirement -> small cut bound
    res = [ad.update(0.5) for _ in range(10)] + [ad.update(0.99) for _ in range(20)]
    assert any(res)


def test_rank_candidates_prefers_test_roc_when_degraded(monkeypatch):
    # Script uses MODELS_DIR on disk; monkeypatch load to a stub so the test
    # does not depend on the (real, honest) on-disk registry.
    import fingraph_sentinel.drift_switcher as ds

    def fake_roc(d: object) -> dict:
        return {
            "baseline-online-xgb": {"val_roc": 0.8937, "test_roc": 0.5967},
            "baseline-online-v3": {"val_roc": 0.8224, "test_roc": 0.7646},
            "baseline-full-xgb": {"val_roc": 0.8906, "test_roc": 0.6456},
        }.get(str(d.name), {})

    monkeypatch.setattr(ds, "load_model_roc", fake_roc)
    rank = rank_candidates("baseline-online-xgb", degraded=True)
    # Velocity beats the serving baseline's observed test ROC, so it should
    # be the top candidate.
    assert rank and rank[0] == "baseline-online-v3"


def test_persist_decision_writes_json(tmp_path):
    from fingraph_sentinel.drift_switcher import SwitchDecision
    d = SwitchDecision(triggered=True, reason="drift",
                       from_model="a", to_model="b")
    path = persist_decision(d, log_dir=tmp_path)
    import json
    data = json.loads(path.read_text())
    assert data["triggered"] and data["to_model"] == "b"
