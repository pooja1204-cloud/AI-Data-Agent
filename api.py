"""
api.py  —  FastAPI wrapper around the DataAgent.

Exposes:
  POST /ask          { question, dataset? }  →  { sql, insight, data, columns }
  GET  /schema       →  { tables: [...] }
  GET  /healthz      →  { status: "ok" }

Run:
  uvicorn api:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager
from typing import Any

import duckdb
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent import DataAgent
from demo import seed_sales_data, seed_trades_data, seed_customers_data


# ---------------------------------------------------------------------------
# App state
# ---------------------------------------------------------------------------

_conn: duckdb.DuckDBPyConnection | None = None
_agent: DataAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _conn, _agent
    _conn = duckdb.connect()
    seed_sales_data(_conn)
    seed_trades_data(_conn)
    seed_customers_data(_conn)
    _agent = DataAgent(_conn)
    print("[api] Agent ready")
    yield
    _conn.close()


app = FastAPI(title="AI Data Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    include_samples: bool = True


class AskResponse(BaseModel):
    question: str
    sql: str
    insight: str
    columns: list[str]
    data: list[dict[str, Any]]
    error: str | None = None


@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    if not _agent:
        raise HTTPException(503, "Agent not initialised")

    result = _agent.ask(req.question, include_samples=req.include_samples)

    if result.error:
        return AskResponse(
            question=result.question,
            sql=result.sql,
            insight="",
            columns=[],
            data=[],
            error=result.error,
        )

    return AskResponse(
        question=result.question,
        sql=result.sql,
        insight=result.insight,
        columns=list(result.data.columns),
        data=result.data.to_dict(orient="records"),
    )


@app.get("/schema")
async def get_schema():
    if not _conn:
        raise HTTPException(503, "DB not ready")

    tables = _conn.execute(
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
    ).fetchall()

    result = {}
    for (table_name,) in tables:
        cols = _conn.execute(
            f"SELECT column_name, data_type FROM information_schema.columns "
            f"WHERE table_name = '{table_name}' ORDER BY ordinal_position"
        ).fetchall()
        row_count = _conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        result[table_name] = {
            "row_count": row_count,
            "columns": [{"name": c, "type": t} for c, t in cols],
        }

    return {"tables": result}


@app.get("/healthz")
async def health():
    return {"status": "ok"}
