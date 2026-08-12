#!/usr/bin/env python3
"""
AI-COMPASS Memory Gateway — Python-native, zero-dependency persistent knowledge server.

Implements the TencentDB Agent Memory v2 API contract so the existing
`memory_hub.py` client works unchanged. Stores everything in local SQLite.

Start:   python tools/memory_gateway.py              (foreground)
         python tools/memory_gateway.py --daemon     (background)
         python tools/ensure_gateway.py              (idempotent auto-start)

Layers:
  L0 Conversation  -> POST /v2/conversation/add, /search   (raw findings)
  L1 Atomic        -> POST /v2/atomic/search               (keyword-searchable facts)
  L2 Scenario      -> POST /v2/scenario/{ls,read,write}    (scene knowledge blocks)
  L3 Core          -> POST /v2/core/{read,write}           (persona / long-term profile)
  Health           -> GET  /health                         (auth-free status)

Storage: ~/.ai-compass/memory/memory.db (SQLite)
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import sqlite3
import subprocess
import sys
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from typing import Any, Dict, List, Optional

DEFAULT_PORT = 8420
DATA_DIR = Path.home() / ".ai-compass" / "memory"
DB_PATH = DATA_DIR / "memory.db"
PID_FILE = DATA_DIR / "gateway.pid"


# ============================================================================
# Database
# ============================================================================

def _get_db() -> sqlite3.Connection:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(str(DB_PATH))
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA busy_timeout=3000")
    return db


def _clamp_limit(val, default=5, max_val=200):
    try:
        v = int(val)
    except (ValueError, TypeError):
        return default
    if v <= 0:
        return default
    return min(v, max_val)


def _init_schema():
    db = _get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            content TEXT NOT NULL,
            ts TEXT NOT NULL DEFAULT (datetime('now')),
            frozen INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_conv_session ON conversations(session_id);
        CREATE INDEX IF NOT EXISTS idx_conv_frozen ON conversations(frozen);

        CREATE TABLE IF NOT EXISTS atomic_memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL DEFAULT 'l1',
            content TEXT NOT NULL,
            background TEXT DEFAULT '',
            score REAL DEFAULT 1.0,
            ts TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS scenarios (
            path TEXT PRIMARY KEY,
            content TEXT NOT NULL DEFAULT '',
            summary TEXT DEFAULT '',
            version INTEGER DEFAULT 1,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS core (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            content TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        INSERT OR IGNORE INTO core (id, content) VALUES (1, '');
    """)
    db.commit()
    db.close()


# ============================================================================
# Distillation pipeline — runs after every conversation add
# ============================================================================

_SIMPLE_WORD = re.compile(r"[a-zA-Z0-9]+")


def _distill_conversation(session_id: str, content: str):
    """Extract atomic facts from raw conversation content using simple keyword
    chunking. The real TencentDB pipeline uses an LLM; we use a lightweight
    heuristic that splits on sentence boundaries and indexes distinct phrases."""
    db = _get_db()
    try:
        sentences = [s.strip() for s in content.replace("\n", ". ").split(".") if len(s.strip()) > 15]
        for sentence in sentences[:8]:
            words = _SIMPLE_WORD.findall(sentence.lower())
            score = min(1.0, len(words) / 20.0)
            db.execute(
                "INSERT INTO atomic_memories (type, content, background, score) VALUES (?, ?, ?, ?)",
                ("l1", sentence[:500], session_id, round(score, 2)),
            )
        db.commit()
    finally:
        db.close()


def _run_pipeline():
    """Process any unfrozen conversations into atomic memories. Called async."""
    try:
        db = _get_db()
        try:
            rows = db.execute(
                "SELECT id, session_id, content FROM conversations WHERE frozen = 0 LIMIT 100"
            ).fetchall()
            if not rows:
                return
            ids = []
            for row in rows:
                _distill_conversation(row["session_id"], row["content"])
                ids.append((row["id"],))
            db.executemany("UPDATE conversations SET frozen = 1 WHERE id = ?", ids)
            db.commit()
        finally:
            db.close()
    except Exception:
        import traceback
        traceback.print_exc(file=sys.stderr)


# ============================================================================
# HTTP Server
# ============================================================================

def _read_json(body: Optional[bytes]) -> Dict[str, Any]:
    if not body:
        return {}
    return json.loads(body.decode("utf-8", "replace"))


def _ok(data: Any = None) -> Dict[str, Any]:
    return {"code": 0, "data": data or {}}


def _err(code: int, message: str) -> Dict[str, Any]:
    return {"code": code, "message": message, "request_id": "gw-local"}


class MemoryGatewayHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, obj: dict, status: int = 200):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _auth_ok(self) -> bool:
        return True

    def do_GET(self):
        if self.path in ("/health", "/healthz"):
            db = _get_db()
            try:
                row = db.execute("SELECT COUNT(*) FROM scenarios").fetchone()
                l2 = row[0] if row else 0
            finally:
                db.close()
            return self._send(_ok({
                "status": "ok",
                "version": "ai-compass-memory/0.1.0",
                "buildInfo": "python-native",
                "vectorStore": "sqlite-local",
                "services": {
                    "pipelineWorker": {"tasksConsumed": 0, "tasksCompleted": 0}
                },
                "scenarios": l2,
            }))
        self._send(_err(404, "not found"))

    def do_POST(self):
        cl = self.headers.get("Content-Length", "0")
        try:
            cl = int(cl)
        except (ValueError, TypeError):
            cl = 0
        body = self.rfile.read(cl) if cl else None
        try:
            payload = _read_json(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._send(_err(400, "invalid json"))

        path = self.path.rstrip("/")

        # --- conversation ---
        if path == "/v2/conversation/add":
            sid = payload.get("session_id", "default")
            msgs = payload.get("messages", [])
            if not msgs:
                return self._send(_err(400, "messages required"))
            db = _get_db()
            ids = []
            try:
                cur = db.cursor()
                for m in msgs:
                    cur.execute(
                        "INSERT INTO conversations (session_id, role, content, ts) VALUES (?, ?, ?, ?)",
                        (sid, m.get("role", "user"), m.get("content", ""),
                         m.get("ts", datetime.datetime.now().isoformat())),
                    )
                    ids.append(cur.lastrowid)
                db.commit()
            finally:
                db.close()
            threading.Thread(target=_run_pipeline, daemon=True).start()
            return self._send(_ok({"accepted_ids": ids, "total_count": len(ids)}))

        if path == "/v2/conversation/search":
            query = payload.get("query", "")
            limit = _clamp_limit(payload.get("limit", 5))
            sid = payload.get("session_id")
            db = _get_db()
            try:
                if sid:
                    rows = db.execute(
                        "SELECT session_id, role, content, ts as updated_at FROM conversations "
                        "WHERE session_id = ? ORDER BY ts DESC LIMIT ?",
                        (sid, limit),
                    ).fetchall()
                else:
                    rows = db.execute(
                        "SELECT session_id, role, content, ts as updated_at FROM conversations "
                        "WHERE content LIKE ? ORDER BY ts DESC LIMIT ?",
                        (f"%{query}%", limit),
                    ).fetchall()
                items = []
                for r in rows:
                    items.append({
                        "session_id": r["session_id"],
                        "role": r["role"],
                        "content": r["content"],
                        "score": 0.5,
                        "updated_at": r["updated_at"],
                    })
            finally:
                db.close()
            return self._send(_ok({"items": items, "total": len(items)}))

        # --- atomic ---
        if path == "/v2/atomic/search":
            query = payload.get("query", "")
            limit = _clamp_limit(payload.get("limit", 5))
            mtype = payload.get("type")
            db = _get_db()
            try:
                sql = "SELECT type, content, background, score, updated_at FROM atomic_memories"
                params: list = []
                conditions = []
                if query:
                    keywords = [w for w in _SIMPLE_WORD.findall(query.lower()) if len(w) > 1]
                    for kw in keywords[:6]:
                        conditions.append("content LIKE ?")
                        params.append(f"%{kw}%")
                if mtype:
                    conditions.append("type = ?")
                    params.append(mtype)
                if conditions:
                    sql += " WHERE " + " AND ".join(conditions)
                sql += " ORDER BY score DESC, updated_at DESC LIMIT ?"
                params.append(limit)
                rows = db.execute(sql, params).fetchall()
                items = []
                for r in rows:
                    items.append({
                        "type": r["type"],
                        "content": r["content"],
                        "score": r["score"],
                        "background": r["background"],
                        "updated_at": r["updated_at"],
                    })
            finally:
                db.close()
            return self._send(_ok({"items": items, "total": len(items)}))

        # --- scenario ---
        if path == "/v2/scenario/ls":
            prefix = payload.get("path_prefix")
            db = _get_db()
            try:
                if prefix:
                    rows = db.execute(
                        "SELECT path, summary, created_at, updated_at FROM scenarios "
                        "WHERE path LIKE ? ORDER BY path",
                        (f"{prefix}%",),
                    ).fetchall()
                else:
                    rows = db.execute(
                        "SELECT path, summary, created_at, updated_at FROM scenarios ORDER BY path"
                    ).fetchall()
                entries = [dict(r) for r in rows]
            finally:
                db.close()
            return self._send(_ok({"entries": entries, "total": len(entries)}))

        if path == "/v2/scenario/read":
            p = payload.get("path")
            if not p:
                return self._send(_err(400, "path required"))
            db = _get_db()
            try:
                row = db.execute(
                    "SELECT * FROM scenarios WHERE path = ?", (p,)
                ).fetchone()
            finally:
                db.close()
            if row:
                return self._send(_ok(dict(row)))
            return self._send(_ok({"path": p, "content": "", "summary": "", "version": 0}))

        if path == "/v2/scenario/write":
            p = payload.get("path")
            content = payload.get("content", "")
            if not p:
                return self._send(_err(400, "path required"))
            content = content[:100000] if content else ""
            summary = (payload.get("summary") or "")[:500]
            db = _get_db()
            try:
                db.execute("BEGIN IMMEDIATE")
                db.execute(
                    "INSERT INTO scenarios (path, content, summary) VALUES (?, ?, ?) "
                    "ON CONFLICT(path) DO UPDATE SET content=excluded.content, "
                    "summary=excluded.summary, updated_at=datetime('now')",
                    (p, content, summary),
                )
                db.commit()
            except sqlite3.Error:
                db.rollback()
            finally:
                db.close()
            return self._send(_ok({"path": p, "status": "written"}))

        # --- core ---
        if path == "/v2/core/read":
            db = _get_db()
            try:
                row = db.execute("SELECT * FROM core WHERE id = 1").fetchone()
            finally:
                db.close()
            return self._send(_ok(dict(row) if row else {"content": ""}))

        if path == "/v2/core/write":
            content = payload.get("content", "")
            db = _get_db()
            try:
                db.execute(
                    "UPDATE core SET content = ?, updated_at = datetime('now') WHERE id = 1",
                    (content,),
                )
                db.commit()
            finally:
                db.close()
            return self._send(_ok({"status": "updated"}))

        return self._send(_err(404, f"unknown route: {path}"))


# ============================================================================
# Server lifecycle
# ============================================================================

def _write_pid():
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PID_FILE.write_text(str(os.getpid()))
    except OSError:
        import traceback
        traceback.print_exc(file=sys.stderr)


    def _remove_pid():
        try:
            PID_FILE.unlink(missing_ok=True)
        except OSError:
            pass


def _is_running():
    if not PID_FILE.exists():
        return False
    try:
        pid = int(PID_FILE.read_text().strip())
        if os.name == 'nt':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x0400, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (ValueError, OSError):
        _remove_pid()
        return False


class _ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def start_server(port: int = DEFAULT_PORT, daemon: bool = False):
    if _is_running():
        print(f"[OK] gateway already running (pid {PID_FILE.read_text().strip()})")
        return 0

    _init_schema()

    if daemon:
        script = Path(__file__).resolve()
        if not script.exists():
            print(f"[FAIL] gateway script not found: {script}", file=sys.stderr)
            return 1
        try:
            subprocess.Popen(
                [sys.executable, str(script)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
            print(f"[FAIL] could not start gateway daemon: {exc}", file=sys.stderr)
            return 1
        for _ in range(10):
            import time
            time.sleep(0.5)
            if _is_running():
                print(f"[OK] gateway daemon started on port {port}")
                return 0
        print(f"[WARN] daemon may not have started; check logs in {DATA_DIR}")
        return 1

    _write_pid()
    server = _ThreadingHTTPServer(("127.0.0.1", port), MemoryGatewayHandler)
    print(f"AI-COMPASS memory gateway listening on http://127.0.0.1:{port}")
    print(f"  data dir: {DATA_DIR}")
    print(f"  pid file: {PID_FILE}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down...")
    finally:
        server.server_close()
        _remove_pid()
    return 0


def main():
    parser = argparse.ArgumentParser(description="AI-COMPASS Memory Gateway")
    parser.add_argument("--daemon", action="store_true", help="Start in background")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"Port (default {DEFAULT_PORT})")
    args = parser.parse_args()
    return start_server(port=args.port, daemon=args.daemon)


if __name__ == "__main__":
    sys.exit(main())
