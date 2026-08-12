###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Persistent job record store for Ray remote execution.

Uses SQLite to track submitted Ray jobs across MCP server restarts.
The database file is often placed on shared storage so it survives process restarts.
"""

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class JobRecord:
    """
    A record of a submitted Ray job.

    Attributes:
        ray_job_id: Ray job ID returned by the Job Submission API
        magpie_task_id: Magpie-internal task ID
        mode_type: "benchmark", "analyze", or "compare"
        ray_cluster: Ray head address used for submission
        config_path: Path to the config JSON on shared storage
        result_path: Path to the result JSON on shared storage
        workspace_dir: Path to the workspace directory on shared storage
        submitted_at: Unix timestamp of submission
        status: Last known status (PENDING, RUNNING, SUCCEEDED, FAILED, STOPPED)
        metadata: Arbitrary key-value metadata (JSON-serialized)
    """
    ray_job_id: str
    magpie_task_id: str
    mode_type: str
    ray_cluster: str
    config_path: str
    result_path: str
    workspace_dir: str = ""
    submitted_at: float = field(default_factory=time.time)
    status: str = "PENDING"
    metadata: Dict[str, Any] = field(default_factory=dict)


class JobStore:
    """
    SQLite-backed persistent store for Ray job records.

    Thread-safe via SQLite's built-in locking.  The database is created
    automatically on first use.
    """

    _CREATE_TABLE = """
    CREATE TABLE IF NOT EXISTS jobs (
        ray_job_id      TEXT PRIMARY KEY,
        magpie_task_id  TEXT NOT NULL,
        mode_type       TEXT NOT NULL,
        ray_cluster     TEXT NOT NULL,
        config_path     TEXT NOT NULL,
        result_path     TEXT NOT NULL,
        workspace_dir   TEXT NOT NULL DEFAULT '',
        submitted_at    REAL NOT NULL,
        status          TEXT NOT NULL DEFAULT 'PENDING',
        metadata        TEXT NOT NULL DEFAULT '{}'
    )
    """

    def __init__(self, db_path: str = ".magpie_jobs.db"):
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self._CREATE_TABLE)
        self._conn.commit()
        logger.info(f"JobStore initialized at {db_path}")

    def add(self, record: JobRecord) -> None:
        """Insert a new job record."""
        import json
        self._conn.execute(
            """INSERT OR REPLACE INTO jobs
               (ray_job_id, magpie_task_id, mode_type, ray_cluster,
                config_path, result_path, workspace_dir,
                submitted_at, status, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.ray_job_id,
                record.magpie_task_id,
                record.mode_type,
                record.ray_cluster,
                record.config_path,
                record.result_path,
                record.workspace_dir,
                record.submitted_at,
                record.status,
                json.dumps(record.metadata),
            ),
        )
        self._conn.commit()

    def get(self, ray_job_id: str) -> Optional[JobRecord]:
        """Get a job record by Ray job ID."""
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE ray_job_id = ?", (ray_job_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def get_by_task_id(self, magpie_task_id: str) -> Optional[JobRecord]:
        """Get a job record by Magpie task ID."""
        row = self._conn.execute(
            "SELECT * FROM jobs WHERE magpie_task_id = ?", (magpie_task_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_jobs(
        self,
        limit: int = 50,
        status: Optional[str] = None,
        mode_type: Optional[str] = None,
    ) -> List[JobRecord]:
        """List job records, most recent first."""
        query = "SELECT * FROM jobs"
        params: List[Any] = []
        conditions = []

        if status:
            conditions.append("status = ?")
            params.append(status)
        if mode_type:
            conditions.append("mode_type = ?")
            params.append(mode_type)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY submitted_at DESC LIMIT ?"
        params.append(limit)

        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    def update_status(self, ray_job_id: str, status: str) -> None:
        """Update the status of a job record."""
        self._conn.execute(
            "UPDATE jobs SET status = ? WHERE ray_job_id = ?",
            (status, ray_job_id),
        )
        self._conn.commit()

    def delete(self, ray_job_id: str) -> None:
        """Delete a job record."""
        self._conn.execute(
            "DELETE FROM jobs WHERE ray_job_id = ?", (ray_job_id,)
        )
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> JobRecord:
        """Convert a SQLite row to a JobRecord."""
        import json
        metadata_str = row["metadata"] if row["metadata"] else "{}"
        try:
            metadata = json.loads(metadata_str)
        except (json.JSONDecodeError, TypeError):
            metadata = {}

        return JobRecord(
            ray_job_id=row["ray_job_id"],
            magpie_task_id=row["magpie_task_id"],
            mode_type=row["mode_type"],
            ray_cluster=row["ray_cluster"],
            config_path=row["config_path"],
            result_path=row["result_path"],
            workspace_dir=row["workspace_dir"],
            submitted_at=row["submitted_at"],
            status=row["status"],
            metadata=metadata,
        )
