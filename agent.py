"""
AI Data Agent for Structured Data
==================================
Converts natural-language questions into SQL using an LLM,
executes them on DuckDB, and returns contextual insights.

Architecture:
  User Question
       ↓
  SchemaInspector   → injects table schema into LLM prompt
       ↓
  LLMQueryPlanner   → generates SQL (Anthropic Claude)
       ↓
  DuckDBExecutor    → runs query, returns DataFrame
       ↓
  InsightGenerator  → LLM summarizes results in plain English
       ↓
  AgentResponse     → structured output (SQL + data + insight)
"""

import os
import json
import textwrap
from dataclasses import dataclass, field
from typing import Any

import duckdb
import pandas as pd
import anthropic


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    question: str
    sql: str
    data: pd.DataFrame
    insight: str
    error: str | None = None

    def display(self):
        print("\n" + "=" * 60)
        print(f"  QUESTION  : {self.question}")
        print("=" * 60)
        if self.error:
            print(f"  ERROR     : {self.error}")
            return
        print(f"\n  SQL QUERY :\n{textwrap.indent(self.sql, '  ')}")
        print(f"\n  RESULTS   :\n{self.data.to_string(index=False)}")
        print(f"\n  INSIGHT   :\n{textwrap.indent(self.insight, '  ')}")
        print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Schema inspector
# ---------------------------------------------------------------------------

class SchemaInspector:
    """Extracts schema metadata from a DuckDB connection."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def get_schema(self) -> str:
        """Returns a DDL-style schema string for all user tables."""
        tables = self.conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
        ).fetchall()

        schema_parts = []
        for (table_name,) in tables:
            cols = self.conn.execute(
                f"SELECT column_name, data_type FROM information_schema.columns "
                f"WHERE table_name = '{table_name}' ORDER BY ordinal_position"
            ).fetchall()
            col_defs = ",\n  ".join(f"{c} {t}" for c, t in cols)
            row_count = self.conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            schema_parts.append(
                f"TABLE: {table_name}  ({row_count:,} rows)\n"
                f"  {col_defs}"
            )

        return "\n\n".join(schema_parts)

    def get_sample(self, table: str, n: int = 3) -> str:
        """Returns a few sample rows for richer context."""
        df = self.conn.execute(f"SELECT * FROM {table} LIMIT {n}").df()
        return df.to_string(index=False)


# ---------------------------------------------------------------------------
# LLM query planner
# ---------------------------------------------------------------------------

class LLMQueryPlanner:
    """Uses Anthropic Claude to translate a natural-language question into SQL."""

    SYSTEM_PROMPT = textwrap.dedent("""
        You are an expert SQL analyst. Your only job is to write a single, correct
        DuckDB SQL query that answers the user's question.

        Rules:
        - Output ONLY the raw SQL — no markdown fences, no explanation.
        - Use DuckDB-compatible syntax (e.g. INTERVAL '30 days', STRFTIME, etc.).
        - Always end with a semicolon.
        - Prefer readable column aliases in UPPER_SNAKE_CASE.
        - If the question is ambiguous, make a reasonable assumption.
    """).strip()

    def __init__(self, client: anthropic.Anthropic, model: str = "claude-opus-4-5"):
        self.client = client
        self.model = model

    def generate_sql(self, question: str, schema: str, samples: str = "") -> str:
        context = f"DATABASE SCHEMA:\n{schema}"
        if samples:
            context += f"\n\nSAMPLE DATA:\n{samples}"

        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=self.SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": f"{context}\n\nQUESTION: {question}"
                }
            ]
        )
        sql = response.content[0].text.strip()
        # Strip accidental markdown fences
        sql = sql.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()
        return sql


# ---------------------------------------------------------------------------
# DuckDB executor
# ---------------------------------------------------------------------------

class DuckDBExecutor:
    """Executes SQL against a DuckDB connection and returns a DataFrame."""

    def __init__(self, conn: duckdb.DuckDBPyConnection):
        self.conn = conn

    def run(self, sql: str) -> pd.DataFrame:
        return self.conn.execute(sql).df()


# ---------------------------------------------------------------------------
# Insight generator
# ---------------------------------------------------------------------------

class InsightGenerator:
    """Uses LLM to produce a plain-English summary of query results."""

    SYSTEM_PROMPT = textwrap.dedent("""
        You are a senior data analyst. Given a business question and query results,
        write 2-4 sentences of sharp, actionable insight. Focus on:
        - The most significant finding (biggest, smallest, trend, anomaly)
        - A concrete implication or recommendation where possible
        - Specific numbers from the results

        Be direct. No fluff. No hedging. Output plain text only.
    """).strip()

    def __init__(self, client: anthropic.Anthropic, model: str = "claude-opus-4-5"):
        self.client = client
        self.model = model

    def generate(self, question: str, sql: str, data: pd.DataFrame) -> str:
        data_preview = data.head(20).to_string(index=False)

        response = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            system=self.SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"QUESTION: {question}\n\n"
                        f"SQL USED:\n{sql}\n\n"
                        f"RESULTS:\n{data_preview}"
                    )
                }
            ]
        )
        return response.content[0].text.strip()


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------

class DataAgent:
    """
    Orchestrates the full pipeline:
      question → schema → SQL → execute → insight → AgentResponse
    """

    def __init__(self, conn: duckdb.DuckDBPyConnection, api_key: str | None = None):
        self.conn = conn
        self.client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
        self.inspector = SchemaInspector(conn)
        self.planner = LLMQueryPlanner(self.client)
        self.executor = DuckDBExecutor(conn)
        self.insight_gen = InsightGenerator(self.client)

    def ask(self, question: str, include_samples: bool = True) -> AgentResponse:
        """Full pipeline: natural language → SQL → data → insight."""
        print(f"[agent] Processing: {question!r}")

        # 1. Gather schema context
        schema = self.inspector.get_schema()

        samples = ""
        if include_samples:
            tables = self.conn.execute(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'main'"
            ).fetchall()
            if tables:
                samples = self.inspector.get_sample(tables[0][0])

        # 2. Generate SQL
        print("[agent] Generating SQL…")
        sql = self.planner.generate_sql(question, schema, samples)
        print(f"[agent] SQL:\n{textwrap.indent(sql, '  ')}")

        # 3. Execute
        try:
            print("[agent] Executing on DuckDB…")
            data = self.executor.run(sql)
        except Exception as exc:
            return AgentResponse(
                question=question,
                sql=sql,
                data=pd.DataFrame(),
                insight="",
                error=str(exc),
            )

        # 4. Generate insight
        print("[agent] Generating insight…")
        insight = self.insight_gen.generate(question, sql, data)

        return AgentResponse(question=question, sql=sql, data=data, insight=insight)
