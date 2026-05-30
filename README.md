# AI-Data-Agent
Data Engineering &amp; Analytics
# AI Data Agent for Structured Data

An LLM-powered analysis agent that converts natural-language questions into SQL,
executes them on DuckDB, and returns contextual insights.

## Architecture

```
User Question
     │
     ▼
SchemaInspector ──── inspects all tables, extracts DDL + sample rows
     │
     ▼
LLMQueryPlanner ──── Claude generates a valid DuckDB SQL query
     │
     ▼
DuckDBExecutor ───── runs the SQL in-memory (or on a file DB)
     │
     ▼
InsightGenerator ─── Claude summarises results in plain English
     │
     ▼
AgentResponse ─────── { question, sql, data: DataFrame, insight }
```

## Quick start

```bash
# Install
pip install -r requirements.txt

# Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run the demo (seeds 3 sample tables, runs 3 questions)
python demo.py

# OR start the REST API
uvicorn api:app --reload --port 8000
```

## REST API

### POST /ask
```json
// Request
{ "question": "Which region had the highest revenue last quarter?" }

// Response
{
  "question": "...",
  "sql": "SELECT region, SUM(amount) ...",
  "insight": "North leads with $2.4M ...",
  "columns": ["region", "total_revenue"],
  "data": [{"region": "North", "total_revenue": 2412300.0}, ...]
}
```

### GET /schema
Returns all table schemas and row counts.

### GET /healthz
Liveness check.

## Key files

| File | Purpose |
|------|---------|
| `agent.py` | Core agent — SchemaInspector, LLMQueryPlanner, DuckDBExecutor, InsightGenerator, DataAgent |
| `demo.py` | Seeds sample data (sales, trades, customers), runs demo questions |
| `api.py` | FastAPI REST wrapper around the agent |
| `requirements.txt` | Python dependencies |

## Extending

**Add your own data**
```python
import duckdb, pandas as pd
from agent import DataAgent

conn = duckdb.connect("mydb.duckdb")
df = pd.read_csv("your_data.csv")
conn.execute("CREATE TABLE my_table AS SELECT * FROM df")

agent = DataAgent(conn)
r = agent.ask("What is the average value per category?")
r.display()
```

**Swap in a different LLM**
Replace the `anthropic.Anthropic` client in `agent.py` with any OpenAI-compatible
client — just change the `generate_sql` and `generate` methods to use your provider.

**Persist to disk**
```python
conn = duckdb.connect("warehouse.duckdb")  # file-backed DB
```

## Sample datasets seeded by demo.py

| Table | Rows | Key columns |
|-------|------|-------------|
| `sales` | 2,000 | order_id, date, product, region, amount, units, rep_name |
| `trades` | 1,500 | trade_id, symbol, side, quantity, price, trader_id, desk |
| `customers` | 800 | customer_id, segment, country, ltv, churn_risk |
