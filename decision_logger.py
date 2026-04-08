"""
decision_logger.py - Persistent logging for validation decisions.
Stores:
- JSONL append log for easy inspection
- SQLite table for structured querying
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any


class PersistentDecisionLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.log_dir / "validation_decisions.jsonl"
        self.sqlite_path = self.log_dir / "validation_decisions.db"
        self._init_db()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    design_id TEXT NOT NULL,
                    component TEXT NOT NULL,
                    passed INTEGER NOT NULL,
                    score REAL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            conn.commit()
        finally:
            conn.close()

    def log_decision(self, payload: Dict[str, Any]) -> None:
        record = dict(payload)
        record.setdefault("timestamp", datetime.utcnow().isoformat() + "Z")
        record.setdefault("component", "unknown")
        record.setdefault("passed", False)
        record.setdefault("design_id", "unknown")

        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")

        conn = sqlite3.connect(self.sqlite_path)
        try:
            conn.execute(
                """
                INSERT INTO decisions (timestamp, design_id, component, passed, score, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record["timestamp"],
                    str(record.get("design_id", "unknown")),
                    str(record.get("component", "unknown")),
                    1 if bool(record.get("passed")) else 0,
                    float(record.get("score")) if record.get("score") is not None else None,
                    json.dumps(record, ensure_ascii=True),
                ),
            )
            conn.commit()
        finally:
            conn.close()
