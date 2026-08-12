#!/usr/bin/env python3
"""
AI-COMPASS Memory Hub Bridge
============================
Client + CLI for the TencentDB Agent Memory *local standalone gateway*
(v2 data-plane API, served by `MemoryCore/src/gateway/server.ts`).

Layers consumed (L0..L3), matching the gateway's team-memory model:
  L0 Conversation  -> POST /v2/conversation/add        (capture reports / run logs)
  L1 Atomic        -> POST /v2/atomic/search            (distilled facts, tool-use recall)
  L2 Scenario      -> POST /v2/scenario/{ls,read,write} (scene knowledge)
  L3 Core          -> POST /v2/core/{read,write}        (persona / long-term profile)

Contracted from `v2-router.ts`:
  - Every /v2 route needs a non-empty `Authorization: Bearer <key>` header and
    `X-Tdai-Service-Id`. In standalone the gateway instanceId is fixed to
    "default" and (unless `TDAI_GATEWAY_API_KEY` is set) any non-empty Bearer
    token is accepted; we send whatever api_key is configured.
  - Response envelope: `{"code":0,"data":{...}}`; any other code raises.
  - `/health` is reachable WITHOUT auth and reports store + pipelineWorker.
  - `/v2/scenario/write` is UPDATE-ONLY: it returns 404 if the file is missing.
  - `/v2/conversation/add` accepts `{session_id, messages:[{role,content}]}` and
    asynchronously mirrors L0 to SQLite + drives the L1/L2/L3 pipeline.

Config (all optional; the gateway is an OPTIONAL sidecar):
  TDAI_MEMORY_ENDPOINT    base URL            (default http://127.0.0.1:8420)
  TDAI_MEMORY_API_KEY     Bearer token        (default "" = unauthenticated)
  TDAI_MEMORY_SERVICE_ID  instance id         (default "default")
  TDAI_MEMORY_TIMEOUT     request timeout s   (default 10)

Usage:
  python tools/memory_hub.py status
  python tools/memory_hub.py capture "MMVQ nwarps=2 gave +24% TG on Qwen3.5-9B" --session tune-qwen
  python tools/memory_hub.py search "MMVQ nwarps RDNA4 fp4" --limit 5
  python tools/memory_hub.py recap "ovalue benchmark"          # markdown context for agent boostrap
  python tools/memory_hub.py scenarios ls
Only the first three commands are imported by `aicompass.py memory`.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import urllib.parse
from http import client as http_client
from pathlib import Path
from typing import Any, Dict, List, Optional

_USER_AGENT = "ai-compass-memory-hub/0.1.0"

DEFAULT_ENDPOINT = os.environ.get("TDAI_MEMORY_ENDPOINT", "http://127.0.0.1:8420")
DEFAULT_API_KEY = os.environ.get("TDAI_MEMORY_API_KEY", "")
DEFAULT_SERVICE_ID = os.environ.get("TDAI_MEMORY_SERVICE_ID", "default")
_DEFAULT_TIMEOUT = 10.0
try:
    _DEFAULT_TIMEOUT = float(os.environ.get("TDAI_MEMORY_TIMEOUT", "10"))
except (TypeError, ValueError):
    pass


class MemoryHubError(RuntimeError):
    """Thrown for non-zero gateway envelopes or transport failures."""

    def __init__(
        self,
        code: Any,
        message: str,
        request_id: Optional[str] = None,
    ) -> None:
        suffix = f" (request_id={request_id})" if request_id else ""
        super().__init__(f"[{code}] {message}{suffix}")
        self.code = code
        self.message = message
        self.request_id = request_id


def _iso(
    body: Dict[str, Any],
    team_id: Optional[str],
    agent_id: Optional[str],
    user_id: Optional[str],
    task_id: Optional[str],
) -> None:
    """Attach the optional team-memory isolation dims from request body."""
    if team_id:
        body["team_id"] = team_id
    if agent_id:
        body["agent_id"] = agent_id
    if user_id:
        body["user_id"] = user_id
    if task_id:
        body["task_id"] = task_id


class MemoryHub:
    """Synchronous, stdlib-only HTTP client for the standalone gateway."""

    def __init__(
        self,
        endpoint: Optional[str] = None,
        api_key: Optional[str] = None,
        service_id: Optional[str] = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self.endpoint = (endpoint or DEFAULT_ENDPOINT).rstrip("/")
        self.api_key = api_key if api_key is not None else DEFAULT_API_KEY
        self.service_id = service_id if service_id is not None else DEFAULT_SERVICE_ID
        self.timeout = timeout

    # --- transport --------------------------------------------------------

    def _split(self):
        p = urllib.parse.urlsplit(self.endpoint)
        scheme = p.scheme or "http"
        host = p.hostname or p.netloc
        if not host:
            raise MemoryHubError(-1, f"invalid endpoint URL: {self.endpoint!r}")
        port = p.port or (443 if scheme == "https" else 80)
        return scheme, host, port

    def _request(self, method: str, path: str, body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        scheme, host, port = self._split()
        conn_cls = http_client.HTTPSConnection if scheme == "https" else http_client.HTTPConnection
        conn = conn_cls(host, port, timeout=self.timeout)
        headers = {
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
            "X-Tdai-Service-Id": self.service_id,
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = json.dumps(body).encode("utf-8") if body is not None else None
        try:
            conn.request(method, path, body=payload, headers=headers)
            resp = conn.getresponse()
            raw = resp.read().decode("utf-8", "replace")
            status = resp.status
        except (OSError, http_client.HTTPException) as exc:
            raise MemoryHubError(-1, f"gateway unreachable at {self.endpoint}: {exc}") from exc
        finally:
            conn.close()

        try:
            envelope = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            envelope = {"code": status, "message": (raw[:500] or "empty response")}

        if isinstance(envelope, dict) and envelope.get("code") == 0:
            return envelope.get("data", {}) or {}

        code = envelope.get("code", status)
        message = envelope.get("message", f"HTTP {status}")
        raise MemoryHubError(code, message, envelope.get("request_id"))

    # --- meta ---------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def is_available(self) -> bool:
        try:
            self.health()
            return True
        except (MemoryHubError, OSError):
            return False

    # --- L0 conversation ----------------------------------------------------

    def add_conversation(
        self,
        session_id: str,
        messages: List[Dict[str, Any]],
        *,
        team_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"session_id": session_id, "messages": messages}
        _iso(body, team_id, agent_id, user_id, task_id)
        return self._request("POST", "/v2/conversation/add", body)

    def search_conversation(
        self,
        query: str,
        *,
        limit: int = 5,
        session_id: Optional[str] = None,
        team_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"query": query, "limit": limit}
        if session_id:
            body["session_id"] = session_id
        _iso(body, team_id, agent_id, user_id, task_id)
        return self._request("POST", "/v2/conversation/search", body)

    # --- L1 atomic ---------------------------------------------------------

    def search_atomic(
        self,
        query: str,
        *,
        limit: int = 5,
        memtype: Optional[str] = None,
        team_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        user_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        body: Dict[str, Any] = {"query": query, "limit": limit}
        if memtype:
            body["type"] = memtype
        _iso(body, team_id, agent_id, user_id, task_id)
        return self._request("POST", "/v2/atomic/search", body)

    # --- L2 scenario --------------------------------------------------------

    def list_scenarios(self, path_prefix: Optional[str] = None, **iso) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        if path_prefix:
            body["path_prefix"] = path_prefix
        _iso(body, iso.get("team_id"), iso.get("agent_id"), iso.get("user_id"), iso.get("task_id"))
        return self._request("POST", "/v2/scenario/ls", body)

    def read_scenario(self, path: str, **iso) -> Dict[str, Any]:
        body: Dict[str, Any] = {"path": path}
        _iso(body, iso.get("team_id"), iso.get("agent_id"), iso.get("user_id"), iso.get("task_id"))
        return self._request("POST", "/v2/scenario/read", body)

    def update_scenario(self, path: str, content: str, summary: Optional[str] = None, **iso) -> Dict[str, Any]:
        """REPLACE the body of an EXISTING scene block (write is update-only)."""
        body: Dict[str, Any] = {"path": path, "content": content}
        if summary:
            body["summary"] = summary
        _iso(body, iso.get("team_id"), iso.get("agent_id"), iso.get("user_id"), iso.get("task_id"))
        return self._request("POST", "/v2/scenario/write", body)

    # --- L3 core / persona --------------------------------------------------

    def read_core(self, **iso) -> Dict[str, Any]:
        body: Dict[str, Any] = {}
        _iso(body, iso.get("team_id"), iso.get("agent_id"), iso.get("user_id"), iso.get("task_id"))
        return self._request("POST", "/v2/core/read", body)

    def write_core(self, content: str, **iso) -> Dict[str, Any]:
        body: Dict[str, Any] = {"content": content}
        _iso(body, iso.get("team_id"), iso.get("agent_id"), iso.get("user_id"), iso.get("task_id"))
        return self._request("POST", "/v2/core/write", body)

    # --- convenience ---------------------------------------------------------

    def capture_text(
        self,
        text: str,
        session_id: str = "ai-compass",
        *,
        role: str = "user",
        **iso,
    ) -> Dict[str, Any]:
        stamp = datetime.datetime.now().isoformat(timespec="seconds")
        messages = [{"role": role, "content": text, "ts": stamp}]
        return self.add_conversation(session_id, messages, **iso)

    def recall_context(self, query: str, *, limit: int = 6) -> str:
        """Return a markdown block of L1 atomic hits ready for context injection.

        Never raises when empty -- callers inject whatever survives.
        """
        lines: List[str] = []
        try:
            data = self.search_atomic(query, limit=min(limit, 20))
        except MemoryHubError as exc:
            lines.append(f"> memory recall unavailable ({exc})")
            return "\n".join(lines)

        items = (data or {}).get("items") or []
        if not items:
            return "> memory recall: no prior findings for this query."
        lines.append("## Recalled agent memory")
        for it in items[:limit]:
            content = it.get("content") or ""
            score = it.get("score")
            when = (it.get("updated_at") or "")[:10]
            meta = " | ".join(x for x in [str(score)[:5] if score is not None else "", when] if x)
            lines.append(f"- {content}\n  (score={meta}, type={it.get('type') or 'l1'})")
        return "\n".join(lines)

    # --- bulk tool helpers ------------------------------------------------------

    def search(self, query: str, *, limit: int = 5, json_out: bool = False) -> Dict[str, Any]:
        """Combined L0 + L1 search. Returns both full envelopes under
        ``atomic`` / ``conversation`` keys."""
        if json_out:
            return {"query": query, "atomic": self.search_atomic(query, limit=limit),
                    "conversation": self.search_conversation(query, limit=limit)}
        return {"query": query, "atomic": self.search_atomic(query, limit=limit),
                "conversation": self.search_conversation(query, limit=limit)}


# --- standalone bridging helpers -------------------------------------------


def recall_context(query: str, *, limit: int = 6,
                   endpoint: Optional[str] = None,
                   api_key: Optional[str] = None,
                   service_id: Optional[str] = None) -> str:
    """Module-level convenience: recall L1 atomic memories as markdown."""
    return MemoryHub(endpoint=endpoint, api_key=api_key,
                     service_id=service_id).recall_context(query, limit=limit)


def publish_report_json(report_path, *, session_id: Optional[str] = None,
                        endpoint: Optional[str] = None,
                        api_key: Optional[str] = None,
                        service_id: Optional[str] = None) -> Dict[str, Any]:
    """Ingest a JSON report (e.g. analyze.py ``analysis.json``) as an L0 conversation.

    Returns the ``/v2/conversation/add`` response envelope.  The gateway's
    async pipeline distills L0 -> L1 -> L2/L3 when a memory/atom/extraction
    path is configured.
    """
    hub = MemoryHub(endpoint=endpoint, api_key=api_key, service_id=service_id)
    rp = Path(report_path)
    try:
        report = json.loads(rp.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise MemoryHubError(-1, f"cannot read report {rp}: {exc}") from exc
    name = rp.stem
    sid = session_id or f"report:{name}"

    summary = report.get("summary") or {}
    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": (
                f"AI-COMPASS optimization report for {name}.\n"
                f"trace: {report.get('trace_file')}\n"
                f"kernels={summary.get('total_kernels')} "
                f"time_ms={summary.get('total_time_ms')} "
                f"gpu_busy_pct={summary.get('estimated_gpu_busy_pct')}\n"
            ),
        }
    ]

    for phase in report.get("phases") or []:
        messages.append({
            "role": "assistant",
            "content": (
                f"phase {phase.get('phase')}: {phase.get('kernel_count')} kernels, "
                f"{phase.get('total_ms')} ms ({phase.get('pct_of_total')}%)"
            ),
        })

    for cat, st in sorted((report.get("category_breakdown") or {}).items(),
                          key=lambda kv: -((kv[1].get("pct", 0) or 0) if isinstance(kv[1], dict) else 0)):
        if isinstance(st, dict):
            messages.append({
                "role": "assistant",
                "content": (
                    f"{cat}: {st.get('count')} kernels, {st.get('total_ms')} ms, "
                    f"{st.get('pct')}%, occ {st.get('avg_occupancy_pct')}%"
                ),
            })

    for b in report.get("bottlenecks") or []:
        messages.append({"role": "assistant", "content": f"bottleneck: {b}"})

    for t in report.get("optimization_targets") or []:
        messages.append({
            "role": "assistant",
            "content": f"target {t.get('target')}: {t.get('suggestion')}",
        })

    return hub.add_conversation(sid, messages)


def _print_atomic_hits(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "  (no hits)"
    out: List[str] = []
    for h in hits[:10]:
        content = (h.get("content") or "")
        if len(content) > 300:
            content = content[:300] + "..."
        out.append(f"  [{h.get('score', '')}] type={h.get('type', 'l1')} :: {content}")
    return "\n".join(out)


def _print_conversation_hits(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return "  (no hits)"
    out: List[str] = []
    for h in hits[:10]:
        content = (h.get("content") or "")
        if len(content) > 200:
            content = content[:200] + "..."
        out.append(f"  [{h.get('score', '')}] {h.get('session_id', h.get('role', ''))}: {content}")
    return "\n".join(out)


def cmd_status(hub: MemoryHub) -> int:
    try:
        h = hub.health()
    except MemoryHubError as exc:
        print(f"[FAIL] Memory gateway unreachable: {exc}")
        print(f"  endpoint: {hub.endpoint}")
        print(f"  Auto-starting Python-native gateway...")
        try:
            import subprocess as _sp, sys as _sys
            this_dir = Path(__file__).resolve().parent
            _sp.run([_sys.executable, str(this_dir / "ensure_gateway.py")],
                    timeout=15, capture_output=True)
            h = hub.health()
            print(f"  [OK] gateway is now live.")
        except (MemoryHubError, OSError, subprocess.SubprocessError):
            print(f"[FAIL] Gateway auto-start failed.")
            print(f"  To start manually: python tools/memory_gateway.py --daemon")
            return 1

    status = h.get("status")
    services = h.get("services") or {}
    worker = services.get("pipelineWorker") or {}
    print("=" * 60)
    print(" TencentDB Agent Memory  (local standalone gateway)")
    print("=" * 60)
    print(f"  endpoint        : {hub.endpoint}")
    print(f"  gateway status  : {status or 'ok'}")
    print(f"  build / version : {h.get('version', h.get('buildInfo', 'n/a'))}")
    print(f"  vectorStore     : {h.get('vectorStore', '')}")
    if worker:
        print(f"  pipeline worker : tasks consumed={worker.get('tasksConsumed')} "
              f"completed={worker.get('tasksCompleted')}")
    try:
        ls = hub.list_scenarios()
        total = (ls or {}).get("total", 0)
        print(f"  L2 scenarios    : {total}")
    except MemoryHubError as exc:
        print(f"  L2 scenarios    : (ls failed: {exc})")
    return 0


def cmd_capture(hub: MemoryHub, args: argparse.Namespace) -> int:
    report_file = Path(args.json_report) if args.json_report else None
    if report_file is not None:
        if not report_file.exists():
            print(f"[FAIL] report file not found: {report_file}")
            return 1
        result = publish_report_json(report_file, session_id=args.session)
        print(f"[OK] captured report {report_file.name} as L0 conversation "
              f"(accepted_ids={result.get('accepted_ids', [])})")
        return 0

    if not args.text:
        print("[FAIL] provide --json-report or a text message")
        return 1

    hub.capture_text(args.text, session_id=args.session or "ai-compass")
    print(f"[OK] captured {len(args.text)} chars -> session '{args.session or 'ai-compass'}'")
    return 0


def cmd_search(hub: MemoryHub, args: argparse.Namespace) -> int:
    try:
        result = hub.search(args.query, limit=args.limit, json_out=args.json)
    except MemoryHubError as exc:
        print(f"[FAIL] search error: {exc}")
        return 1

    if args.json:
        print(json.dumps(result, indent=2))
        return 0
    print(f"== L1 atomic memories for '{args.query}' ==")
    print(_print_atomic_hits(result.get("atomic", {}).get("items", [])))
    print(f"== L0 conversations for '{args.query}' ==")
    print(_print_conversation_hits(result.get("conversation", {}).get("items", [])))
    return 0


def cmd_recap(hub: MemoryHub, args: argparse.Namespace) -> int:
    try:
        text = hub.recall_context(args.query, limit=args.limit)
        print(text)
    except MemoryHubError as exc:
        print(f"[FAIL] {exc}")
        return 1
    return 0


def cmd_scenarios(hub: MemoryHub, args: argparse.Namespace) -> int:
    sub = args.scenario_command
    try:
        if sub == "ls":
            data = hub.list_scenarios(args.prefix)
            entries = (data or {}).get("entries") or []
            print(f"L2 scenarios: {len(entries)}")
            for e in entries:
                print(f"  {e.get('path')}" + (f"  -- {e.get('summary')}" if e.get('summary') else ""))
            return 0
        if sub == "read":
            data = hub.read_scenario(args.path)
            content = (data or {}).get("content")
            print(content if content else f"(no scene at {args.path})")
            return 0
        if sub == "write":
            if not args.content:
                print("[FAIL] needs --content")
                return 1
            hub.update_scenario(args.path, args.content, summary=args.summary)
            print(f"[OK] updated scenario {args.path}")
            return 0
    except MemoryHubError as exc:
        print(f"[FAIL] {exc}")
        return 1
    print("[ERR] unknown subcommand; use ls|read|write")
    return 2


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="TencentDB Agent Memory bridge for AI-COMPASS")
    parser.add_argument("--endpoint", default=None, help="gateway endpoint (env TDAI_MEMORY_ENDPOINT)")
    parser.add_argument("--api-key", default=None, help="Bearer token (env TDAI_MEMORY_API_KEY)")
    parser.add_argument("--service-id", default=None, help="instance id (env TDAI_MEMORY_SERVICE_ID)")
    parser.add_argument("--timeout", type=int, default=10, choices=range(1, 301), help="request timeout s")
    sub = parser.add_subparsers(dest="command", required=True)

    p_status = sub.add_parser("status", help="show gateway health + asset counts")

    p_capture = sub.add_parser("capture", help="capture text or a JSON report into L0 memory")
    p_capture.add_argument("--text", default=None, help="short text to remember")
    p_capture.add_argument("--json-report", default=None, help="path to analysis.json/ --ingest a whole report")
    p_capture.add_argument("--session", default=None, help="session id to attach the capture to")

    p_search = sub.add_parser("search", help="search L1 + L0 memory")
    p_search.add_argument("--query", "-q", required=True)
    p_search.add_argument("--limit", type=int, default=5)
    p_search.add_argument("--json", action="store_true", help="dump raw JSON")

    p_recap = sub.add_parser("recap", help="render recalled memory as markdown for agent bootstrap")
    p_recap.add_argument("--query", "-q", required=True)
    p_recap.add_argument("--limit", type=int, default=6)

    p_scn = sub.add_parser("scenarios", help="L2 scene blocks")
    scn_sub = p_scn.add_subparsers(dest="scenario_command")
    p_ls = scn_sub.add_parser("ls")
    p_ls.add_argument("--prefix", default=None)
    p_rd = scn_sub.add_parser("read")
    p_rd.add_argument("path")
    p_wr = scn_sub.add_parser("write")
    p_wr.add_argument("path")
    p_wr.add_argument("--content", default=None)
    p_wr.add_argument("--summary", default=None)

    p_persona = sub.add_parser("persona", help="L3 persona (core memory)")
    persona_sub = p_persona.add_subparsers(dest="persona_command")
    persona_sub.add_parser("read")
    p_pw = persona_sub.add_parser("write")
    p_pw.add_argument("--content", required=True)

    args = parser.parse_args(argv)

    hub = MemoryHub(endpoint=args.endpoint, api_key=args.api_key,
                    service_id=args.service_id, timeout=args.timeout)

    if not hub.is_available():
        print(f"[WARN] gateway not reachable at {hub.endpoint}")
        print("       start it or set TDAI_MEMORY_ENDPOINT; continuing anyway.")

    if args.command == "status":
        return cmd_status(hub)
    if args.command == "capture":
        return cmd_capture(hub, args)
    if args.command == "search":
        return cmd_search(hub, args)
    if args.command == "recap":
        return cmd_recap(hub, args)
    if args.command == "scenarios":
        return cmd_scenarios(hub, args)
    if args.command == "persona":
        if args.persona_command == "read":
            try:
                data = hub.read_core()
                print((data or {}).get("content") or "(no core memory yet)")
                return 0
            except MemoryHubError as exc:
                print(f"[FAIL] {exc}")
                return 1
        elif args.persona_command == "write":
            try:
                hub.write_core(args.content)
                print("[OK] persona updated")
                return 0
            except MemoryHubError as exc:
                print(f"[FAIL] {exc}")
                return 1
        else:
            parser.print_help()
            return 1
    parser.print_help()
    return 0


__version__ = "0.1.0"

if __name__ == "__main__":
    sys.exit(main())