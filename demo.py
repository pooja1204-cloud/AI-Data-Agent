"""
demo.py  —  Run the AI Data Agent on sample financial datasets.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python demo.py
"""

import random
from datetime import date, timedelta

import duckdb
import pandas as pd

from agent import DataAgent


# ---------------------------------------------------------------------------
# Seed realistic sample data
# ---------------------------------------------------------------------------

def seed_sales_data(conn: duckdb.DuckDBPyConnection, n: int = 2000):
    random.seed(42)
    products = ["Analytics Pro", "Data Vault", "Cloud Bridge", "Stream Engine", "Query Suite"]
    regions   = ["North", "South", "East", "West"]
    reps      = ["Alice Chen", "Bob Patel", "Carla Ruiz", "David Kim", "Eva Nair"]

    rows = []
    base = date(2024, 1, 1)
    for i in range(n):
        rows.append({
            "order_id":  f"ORD-{i+1:05d}",
            "date":      base + timedelta(days=random.randint(0, 364)),
            "product":   random.choice(products),
            "region":    random.choice(regions),
            "amount":    round(random.uniform(500, 15000), 2),
            "units":     random.randint(1, 20),
            "rep_name":  random.choice(reps),
        })
    df = pd.DataFrame(rows)
    conn.execute("CREATE OR REPLACE TABLE sales AS SELECT * FROM df")
    print(f"[seed] Created 'sales' table with {n} rows")


def seed_trades_data(conn: duckdb.DuckDBPyConnection, n: int = 1500):
    random.seed(7)
    symbols  = ["AAPL", "MSFT", "JPM", "GS", "BAC", "TSLA", "NVDA", "AMZN"]
    desks    = ["Equity", "Fixed Income", "FX", "Rates"]
    traders  = [f"T{i:03d}" for i in range(1, 21)]

    rows = []
    base = date(2024, 1, 1)
    for i in range(n):
        rows.append({
            "trade_id":        f"TRD-{i+1:06d}",
            "symbol":          random.choice(symbols),
            "side":            random.choice(["BUY", "SELL"]),
            "quantity":        random.randint(100, 50000),
            "price":           round(random.uniform(10, 500), 2),
            "trader_id":       random.choice(traders),
            "desk":            random.choice(desks),
            "settlement_date": base + timedelta(days=random.randint(0, 364)),
        })
    df = pd.DataFrame(rows)
    conn.execute("CREATE OR REPLACE TABLE trades AS SELECT * FROM df")
    print(f"[seed] Created 'trades' table with {n} rows")


def seed_customers_data(conn: duckdb.DuckDBPyConnection, n: int = 800):
    random.seed(21)
    segments  = ["Enterprise", "Mid-Market", "SMB"]
    countries = ["USA", "UK", "India", "Germany", "Singapore"]
    risks     = ["low", "medium", "high"]

    rows = []
    base = date(2022, 1, 1)
    for i in range(n):
        seg = random.choice(segments)
        rows.append({
            "customer_id": f"CUST-{i+1:05d}",
            "name":        f"Company {i+1}",
            "segment":     seg,
            "country":     random.choice(countries),
            "join_date":   base + timedelta(days=random.randint(0, 730)),
            "ltv":         round(random.uniform(
                               5000 if seg == "Enterprise" else 1000,
                               200000 if seg == "Enterprise" else 50000
                           ), 2),
            "churn_risk":  random.choices(
                               risks,
                               weights=[50, 30, 20] if seg == "Enterprise" else [30, 30, 40]
                           )[0],
        })
    df = pd.DataFrame(rows)
    conn.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM df")
    print(f"[seed] Created 'customers' table with {n} rows")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

DEMO_QUESTIONS = [
    # Sales
    ("sales",     "What was total revenue and number of orders by region?"),
    ("sales",     "Who are the top 3 sales reps by total amount in Q1 2024?"),
    ("sales",     "Which product generates the most revenue per unit sold?"),
    # Trades
    ("trades",    "What is total notional value by desk, sorted descending?"),
    ("trades",    "Which symbol has the highest buy-to-sell ratio?"),
    # Customers
    ("customers", "What percentage of customers in each segment have high churn risk?"),
    ("customers", "Which country has the highest average LTV?"),
]


def main():
    # In-memory DuckDB — swap for duckdb.connect("mydb.duckdb") to persist
    conn = duckdb.connect()

    print("=" * 60)
    print("  Seeding sample datasets…")
    print("=" * 60)
    seed_sales_data(conn)
    seed_trades_data(conn)
    seed_customers_data(conn)

    agent = DataAgent(conn)

    print("\n" + "=" * 60)
    print("  Running demo questions…")
    print("=" * 60)

    for _table, question in DEMO_QUESTIONS[:3]:   # Run first 3 to keep demo quick
        response = agent.ask(question)
        response.display()


if __name__ == "__main__":
    main()
