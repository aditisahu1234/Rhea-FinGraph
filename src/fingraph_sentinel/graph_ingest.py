"""Ingest Rhea FinGraph transaction data into Neo4j as a heterogeneous graph.

Graph schema:
  Nodes:  :Customer (id)
          :Merchant (id, mcc_code, fraud_rate)
          :Card     (id)
  Edges:  (Customer)-[:PURCHASED {amount, time, channel, is_fraud}]->(Merchant)
          (Customer)-[:HAS_CARD]->(Card)
          (Card)-[:SWIPED_AT {amount, time, channel, is_fraud}]->(Merchant)

Memory-efficient: streams edges from parquet in chunks.
Transaction-safe: small tx_batch to avoid Neo4j commit failures on 24M edges.

Run:  python -m fingraph_sentinel.graph_ingest
      make ingest-graph
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import polars as pl
from neo4j import GraphDatabase


def _driver(url: str, user: str, password: str):
    return GraphDatabase.driver(url, auth=(user, password))


def create_schema(tx):
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Customer) REQUIRE c.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Merchant) REQUIRE m.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (ca:Card) REQUIRE ca.id IS UNIQUE")


# ── Cypher ──────────────────────────────────────────────────────────────────

_CYPHER_CUSTOMER = "UNWIND $rows AS row MERGE (c:Customer {id: row.customer_id})"
_CYPHER_MERCHANT = """
    UNWIND $rows AS row
    MERGE (m:Merchant {id: row.merchant_id})
      ON CREATE SET m.mcc_code = row.mcc, m.fraud_rate = row.fraud_rate
      ON MATCH  SET m.mcc_code = row.mcc, m.fraud_rate = row.fraud_rate
"""
_CYPHER_CARD = "UNWIND $rows AS row MERGE (ca:Card {id: row.card_id})"

_CYPHER_HAS_CARD = """
    UNWIND $rows AS row
    MATCH (c:Customer {id: row.customer_id})
    MATCH (ca:Card {id: row.card_id})
    MERGE (c)-[:HAS_CARD]->(ca)
"""

_CYPHER_PURCHASED = """
    UNWIND $rows AS row
    MATCH (c:Customer {id: row.customer_id})
    MATCH (m:Merchant {id: row.merchant_id})
    CREATE (c)-[:PURCHASED {
        transaction_id: row.transaction_id,
        amount: row.amount,
        time: row.time,
        channel: row.channel,
        is_fraud: row.is_fraud
    }]->(m)
"""


# ── node ingestion ──────────────────────────────────────────────────────────

_EDGE_COLS = [
    "transaction_id", "customer_id", "merchant_id", "card_id",
    "amount", "event_time", "payment_channel", "is_fraud",
]


def ingest_nodes(session, parquet_files: list[Path], tx_batch: int):
    customers: set[str] = set()
    cards: set[str] = set()
    merchant_data: dict[str, dict] = {}

    for f in parquet_files:
        lf = pl.scan_parquet(f)
        customers.update(lf.select("customer_id").unique().collect()["customer_id"].to_list())
        cards.update(lf.select("card_id").unique().collect()["card_id"].to_list())
        mf = (
            lf.select(["merchant_id", "merchant_category_code", "is_fraud"])
            .group_by("merchant_id")
            .agg(
                pl.col("merchant_category_code").first().alias("mcc"),
                pl.col("is_fraud").mean().alias("fraud_rate"),
            )
            .collect()
        )
        for row in mf.to_dicts():
            mid = row["merchant_id"]
            if mid not in merchant_data:
                merchant_data[mid] = row
            else:
                old = merchant_data[mid]
                merchant_data[mid] = {"merchant_id": mid, "mcc": row["mcc"] or old["mcc"],
                                       "fraud_rate": (row["fraud_rate"] + old["fraud_rate"]) / 2}
        print(f"  [scan] {f.name} done", flush=True)

    # Customers
    rows = [{"customer_id": c} for c in customers]
    t0 = time.time()
    for i in range(0, len(rows), tx_batch):
        session.execute_write(lambda tx, r=rows[i:i+tx_batch]: tx.run(_CYPHER_CUSTOMER, rows=r))
    print(f"  [Customer] {len(rows):,} nodes in {time.time()-t0:.1f}s", flush=True)

    # Merchants
    rows = [
        {
            "merchant_id": m["merchant_id"],
            "mcc": m.get("mcc", ""),
            "fraud_rate": m.get("fraud_rate", 0.0),
        }
        for m in merchant_data.values()
    ]
    t0 = time.time()
    for i in range(0, len(rows), tx_batch):
        session.execute_write(lambda tx, r=rows[i:i+tx_batch]: tx.run(_CYPHER_MERCHANT, rows=r))
    print(f"  [Merchant] {len(rows):,} nodes in {time.time()-t0:.1f}s", flush=True)

    # Cards
    rows = [{"card_id": c} for c in cards]
    t0 = time.time()
    for i in range(0, len(rows), tx_batch):
        session.execute_write(lambda tx, r=rows[i:i+tx_batch]: tx.run(_CYPHER_CARD, rows=r))
    print(f"  [Card] {len(rows):,} nodes in {time.time()-t0:.1f}s", flush=True)


# ── edge ingestion ──────────────────────────────────────────────────────────

def ingest_edges(session, parquet_files: list[Path], tx_batch: int):
    # HAS_CARD
    hc_pairs: set[tuple[str, str]] = set()
    for f in parquet_files:
        pairs = pl.scan_parquet(f).select(["customer_id", "card_id"]).unique().collect()
        for row in pairs.to_dicts():
            hc_pairs.add((row["customer_id"], row["card_id"]))
    hc_list = [{"customer_id": c, "card_id": ca} for c, ca in hc_pairs]
    t0 = time.time()
    for i in range(0, len(hc_list), tx_batch):
        session.execute_write(lambda tx, r=hc_list[i:i+tx_batch]: tx.run(_CYPHER_HAS_CARD, rows=r))
    print(f"  [HAS_CARD] {len(hc_list):,} edges in {time.time()-t0:.1f}s", flush=True)
    del hc_pairs, hc_list

    # PURCHASED — stream from parquet, small sub-batches
    total_all = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in parquet_files)
    total_inserted = 0
    t0 = time.time()

    for f in parquet_files:
        lf = pl.scan_parquet(f)
        file_total = lf.select(pl.len()).collect().item()
        offset = 0
        while offset < file_total:
            rows = (
                lf.slice(offset, tx_batch)
                .select(_EDGE_COLS)
                .rename({"event_time": "time", "payment_channel": "channel"})
                .collect()
            )
            if rows.is_empty():
                break
            batch = rows.to_dicts()
            session.execute_write(lambda tx, r=batch: tx.run(_CYPHER_PURCHASED, rows=r))
            total_inserted += len(batch)
            offset += len(batch)
            elapsed = time.time() - t0
            rate = total_inserted / elapsed if elapsed > 0 else 0
            pct = total_inserted / total_all * 100
            if total_inserted % (tx_batch * 20) < tx_batch or total_inserted >= total_all:
                print(
                    f"  [PURCHASED] {total_inserted:,}/{total_all:,}"
                    f" ({pct:.0f}%) — {rate:,.0f} edges/s",
                    flush=True,
                )

    print(f"  [PURCHASED] done in {time.time()-t0:.1f}s — {total_inserted:,} edges", flush=True)


def print_stats(session):
    print("\n=== GRAPH STATS ===")
    for label in ("Customer", "Merchant", "Card"):
        r = session.run(f"MATCH (n:{label}) RETURN count(n) AS c").single()
        print(f"  {label} nodes: {r['c']:,}")
    for rel in ("PURCHASED", "HAS_CARD"):
        r = session.run(f"MATCH ()-[r:{rel}]->() RETURN count(r) AS c").single()
        print(f"  {rel} edges:  {r['c']:,}")
    r = session.run("MATCH ()-[r {is_fraud: 1}]->() RETURN count(r) AS c").single()
    print(f"  Fraud edges: {r['c']:,}")
    # Sample fraud ring
    for r in session.run("""
        MATCH (c:Customer)-[r:PURCHASED]->(m:Merchant)
        WHERE r.is_fraud = 1 AND m.fraud_rate > 0.05
        RETURN c.id AS cust, m.id AS merch, m.fraud_rate AS rate
        LIMIT 5
    """):
        print(f"    {r['cust']} -> {r['merch']} (fraud_rate={r['rate']:.4f})")


def main():
    parser = argparse.ArgumentParser(description="Ingest transactions into Neo4j fraud graph.")
    parser.add_argument("--splits", nargs="+", default=["train"])
    parser.add_argument("--all-splits", action="store_true", help="Ingest train + val + test")
    parser.add_argument(
        "--tx-batch",
        type=int,
        default=5000,
        help="Rows per Neo4j transaction (default 5000)",
    )
    parser.add_argument("--url", default="bolt://localhost:7687")
    parser.add_argument("--user", default="neo4j")
    parser.add_argument(
        "--password",
        default=None,
        help="Neo4j password (defaults to FINGRAPH_NEO4J_PASSWORD / NEO4J_PASSWORD from settings/.env)",
    )
    args = parser.parse_args()

    if args.password is None:
        try:
            from fingraph_sentinel.config import get_settings

            args.password = get_settings().neo4j_password
        except Exception:  # noqa: BLE001 - fall back to the historical default
            args.password = "change-me-local-only"

    if args.all_splits:
        args.splits = ["train", "val", "test"]

    base = Path("data/processed/ibm_full")
    pmap = {"train": "train.parquet", "val": "validation.parquet", "test": "test.parquet"}
    files = [base / pmap[s] for s in args.splits if s in pmap]
    if not files:
        print("No valid splits.")
        return

    print(f"[ingest] splits: {[f.name for f in files]}", flush=True)

    driver = _driver(args.url, args.user, args.password)
    session = driver.session(database="neo4j")

    print("[schema] constraints + indexes ...", flush=True)
    session.execute_write(create_schema)

    print("[nodes] unique entities ...", flush=True)
    ingest_nodes(session, files, args.tx_batch)

    total = sum(pl.scan_parquet(f).select(pl.len()).collect().item() for f in files)
    print(
        f"[edges] {total:,} transactions as PURCHASED edges"
        f" (tx_batch={args.tx_batch}) ...",
        flush=True,
    )
    ingest_edges(session, files, args.tx_batch)

    print_stats(session)
    session.close()
    driver.close()
    print("\n[done] Open http://localhost:7474 to explore.")


if __name__ == "__main__":
    main()
