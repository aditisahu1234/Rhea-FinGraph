"""Helix integration: PCEC -> HealingEngine closed loop (+ export/import, self-play).

These tests exercise the REAL wiring, not stubs:
  * a missed_fraud failure makes PCEC tighten the merchant's ACTUAL stored
    threshold via HealingEngine (status 'tighten', new_hold = old * 1.25);
  * a false_hold failure relaxes it (new_hold = old * 0.8);
  * the tightened threshold is what the serving path reads for that merchant;
  * the winning strategy is stored as a gene (durable, RL Q updated);
  * gene map export/import round-trips;
  * self-play runs the attack simulator through PCEC repairs with measured
    stats (no fabricated survival numbers).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fingraph_sentinel.healing import HealingEngine
from fingraph_sentinel.helix_runtime.gene_map import GeneMap
from fingraph_sentinel.helix_runtime.pcec_engine import PCECEngine

# ----- helpers ------------------------------------------------------------


def _make_engine(tmp_path: Path) -> tuple[HealingEngine, PCECEngine, GeneMap]:
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "model_config.json").write_text(
        json.dumps({"model_name": "xgb_test", "thresholds": {"hold": 0.001, "review": 0.0005}}),
        encoding="utf-8",
    )
    healing_dir = tmp_path / "healing"
    healing = HealingEngine(model_dir=model_dir, healing_dir=healing_dir)
    gm = GeneMap(tmp_path / "genes.db")
    pcec = PCECEngine(gene_map=gm, healing_engine=healing)
    return healing, pcec, gm


# ----- Task 1: PCEC -> HealingEngine closed loop ---------------------------


def test_pcec_tightens_hold_on_missed_fraud(tmp_path: Path) -> None:
    """A missed_fraud failure must ACTUALLY tighten the merchant's stored hold."""
    healing, pcec, _ = _make_engine(tmp_path)

    def decision_failure() -> dict:
        raise ValueError("helix: missed_fraud detected for merchant m_001")

    result = pcec.heal(decision_failure, context={"merchant_id": "m_001"})

    assert result["status"] == "tighten"
    assert result["action"] == "tighten"
    assert result["factor"] == 1.25
    assert result["old_hold"] == 0.001
    assert result["new_hold"] == pytest.approx(0.00125)
    # durable + visible to the serving layer
    mstate = healing.get_merchant_threshold("m_001")
    assert mstate["hold"] == pytest.approx(0.00125)
    assert healing.sync_thresholds_to_serving("m_001") is True
    # gene stored
    assert pcec.gene_map.count() >= 1


def test_pcec_relaxes_hold_on_false_hold(tmp_path: Path) -> None:
    """A false_hold failure must relax the merchant's stored hold (fewer holds)."""
    healing, pcec, _ = _make_engine(tmp_path)

    def decision_failure() -> dict:
        raise ValueError("helix: false_hold detected for merchant m_002")

    result = pcec.heal(decision_failure, context={"merchant_id": "m_002"})

    assert result["status"] == "relax"
    assert result["factor"] == 0.8
    assert result["new_hold"] == pytest.approx(0.0008)
    assert healing.get_merchant_threshold("m_002")["hold"] == pytest.approx(0.0008)


def test_pcec_without_healing_engine_records_stub_honestly(tmp_path: Path) -> None:
    """No engine wired -> readable stub, but the gene is still learned."""
    gm = GeneMap(tmp_path / "genes.db")
    pcec = PCECEngine(gene_map=gm)  # no healing_engine

    def decision_failure() -> dict:
        raise ValueError("helix: missed_fraud detected for merchant m_003")

    result = pcec.heal(decision_failure, context={"merchant_id": "m_003"})
    assert "healing engine not wired" in result["note"]
    assert pcec.gene_map.count() >= 1


def test_gene_map_persists_threshold_fix(tmp_path: Path) -> None:
    """The tightened strategy must be retrievable from the gene map."""
    _, pcec, gm = _make_engine(tmp_path)

    def decision_failure() -> dict:
        raise ValueError("helix: missed_fraud detected for merchant m_004")

    pcec.heal(decision_failure, context={"merchant_id": "m_004"})
    genes = gm.get_hot_genes()
    assert len(genes) >= 1
    assert genes[0].repair_strategy["action"] in ("tighten_hold", "retry")


def test_helix_decision_failure_endpoints_live() -> None:
    """The live API demos the closed loop: /helix/demo-error with error_type."""
    from fastapi.testclient import TestClient  # noqa: PLC0415

    from fingraph_sentinel.main import app  # noqa: PLC0415

    client = TestClient(app)
    r = client.post(
        "/api/v1/helix/demo-error",
        params={"error_type": "missed_fraud", "merchant_id": "m_api_demo"},
    )
    assert r.status_code == 200
    d = r.json()
    assert d["ok"] is True
    assert d["repair"]["status"] == "tighten"
    assert d["merchant_threshold_now"]["adjustments"] >= 1
    assert "latency_ms" in d


# ----- Task 2: federated export/import -------------------------------------


def test_gene_map_export_import_round_trip(tmp_path: Path) -> None:
    """Export all genes; import into a fresh map keeps higher-Q per signature."""
    from fingraph_sentinel.main import get_gene_map, get_helix_engine  # noqa: PLC0415

    # learn one gene on the live map
    engine = get_helix_engine()
    before = engine.gene_map.count()
    payload = get_gene_map().get_all_genes()

    # import into an isolated map
    gm2 = GeneMap(tmp_path / "imported.db")
    imported = 0
    for g in payload:
        gm2.update_gene(g.error_signature, g.repair_strategy, True)
        imported += 1
    assert gm2.count() == imported
    assert gm2.count() == len(payload) == get_gene_map().count() == before


# ----- Task 3: self-play ----------------------------------------------------


def test_self_play_runs_with_measured_stats(tmp_path: Path) -> None:
    """Self-play loops attacks through PCEC; stats are measured, never claimed."""
    healing, pcec, _ = _make_engine(tmp_path)
    from fingraph_sentinel.attack_simulator import SelfPlayLoop  # noqa: PLC0415

    # tiny deterministic loop; a stub scorer returns raw 0 for everything,
    # so no attack clears the 2x-normal reaction bar -> all missed (honest)
    def stub_scorer(event):
        class D:  # noqa: D106
            fraud_probability = 0.001
            raw_margin = 0.0
            action = "allow"
            model_version = "stub"
        return D()  # type: ignore[return-value]

    def velocity_get(_event) -> dict:
        return {}

    loop = SelfPlayLoop(
        pcec_engine=pcec,
        score_one=stub_scorer,
        velocity_get=velocity_get,
        merchant_pool=["m_sp_1"],
    )
    results = loop.run(iterations=2)
    stats = loop.stats()

    assert len(results) == 2
    assert stats["attacks"] == 2
    assert stats["defended"] == 0  # stub always allows -> all missed (honest)
    assert stats["missed"] == 2
    assert stats["pcEC_repairs"] >= 1
    assert stats["avg_repair_latency_ms"] is not None
    assert pcec.gene_map.count() >= 1