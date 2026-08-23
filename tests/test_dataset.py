from pathlib import Path

import polars as pl

from fingraph_sentinel.dataset import normalize_ibm_transactions, write_temporal_splits


def _write_ibm_fixture(path: Path) -> None:
    rows = [
        {
            "User": "alice",
            "Card": "1",
            "Year": 2023,
            "Month": 1,
            "Day": day,
            "Time": "10:30",
            "Amount": f"${100 + day}.00",
            "Use Chip": "Chip Transaction",
            "Merchant Name": "merchant-a",
            "Merchant City": "Austin",
            "Merchant State": "TX",
            "Zip": "78701",
            "MCC": "5411",
            "Errors?": None,
            "Is Fraud?": "Yes" if day in {5, 10} else "No",
        }
        for day in range(1, 11)
    ]
    pl.DataFrame(rows).write_csv(path)


def test_normalize_ibm_transactions(tmp_path: Path) -> None:
    source = tmp_path / "ibm.csv"
    _write_ibm_fixture(source)

    result = normalize_ibm_transactions(source).collect()

    assert result.columns == [
        "transaction_id",
        "event_time",
        "customer_id",
        "card_id",
        "merchant_id",
        "merchant_category_code",
        "amount",
        "currency",
        "merchant_city",
        "merchant_state",
        "merchant_zip",
        "payment_channel",
        "payment_error",
        "is_fraud",
    ]
    assert result["amount"].to_list()[0] == 101.0
    assert result["is_fraud"].sum() == 2
    assert result["card_id"].to_list()[0] == "alice::1"


def test_temporal_split_has_no_time_overlap(tmp_path: Path) -> None:
    source = tmp_path / "ibm.csv"
    output = tmp_path / "processed"
    _write_ibm_fixture(source)

    manifest = write_temporal_splits(source, output)

    train = pl.read_parquet(output / "train.parquet")
    validation = pl.read_parquet(output / "validation.parquet")
    test = pl.read_parquet(output / "test.parquet")

    assert manifest["splits"]["train"]["rows"] > 0
    assert train["event_time"].max() < validation["event_time"].min()
    assert validation["event_time"].max() < test["event_time"].min()
