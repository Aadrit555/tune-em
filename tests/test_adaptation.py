"""Unit tests for online adaptation and train/eval leakage prevention."""

import pytest
from anydecision.adaptation.buffer import ObservationBuffer
from anydecision.adaptation.online import OnlineAdapter, create_calibration_splits


def test_data_leakage_protection():
    buf = ObservationBuffer(max_capacity=100)
    buf.register_evaluation_split(["eval_q1", "eval_q2"])

    # Adding a non-eval question is allowed
    buf.add("train_q1", {"yes": 0.9, "no": 0.1}, "yes", "yes")
    assert buf.count() == 1

    # Attempting to add an evaluation question raises ValueError immediately
    with pytest.raises(ValueError, match="Data Leakage Violation"):
        buf.add("eval_q1", {"yes": 0.8, "no": 0.2}, "yes", "yes")


def test_create_calibration_splits():
    data = [{"id": f"q_{i}", "label": i % 2} for i in range(100)]
    splits = create_calibration_splits(data, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=42)

    assert len(splits["train_calibration"]) == 60
    assert len(splits["validation"]) == 20
    assert len(splits["test"]) == 20

    # Ensure partitions are completely disjoint
    train_ids = {item["id"] for item in splits["train_calibration"]}
    val_ids = {item["id"] for item in splits["validation"]}
    test_ids = {item["id"] for item in splits["test"]}

    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)


def test_online_adapter_observe():
    adapter = OnlineAdapter(refit_interval=5)
    for i in range(5):
        updated = adapter.observe(
            question_id=f"q_{i}",
            probabilities={"catA": 0.7, "catB": 0.3},
            prediction="catA",
            label="catA" if i % 2 == 0 else "catB",
        )
    assert updated is True
    assert adapter.updates_count == 1
