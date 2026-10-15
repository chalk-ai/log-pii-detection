"""Backend for /demos/log-pii-detection, served from a Chalk ScalingGroup.

- ``GET /api/tail?after=N``: the shared live tail. One background loop scans a few synthetic log
  lines every couple of seconds; every visitor reads the same ring buffer, so load stays flat.
- ``POST /api/scan``: scan one pasted line. Nothing is stored or logged: the access log is off and
  the text exists only for the duration of the request.
- ``POST /api/sql``: run one of two fixed ChalkSQL queries over a log table, calling the deployed
  ``function.log-pii-redact`` through ``catalog_call``. Visitors pick a preset; they never send SQL.
- ``GET /api/metrics``: precision/recall measured offline on the held-out set, plus live latency.
"""

from __future__ import annotations

import asyncio
import base64
import collections
import io
import json
import os
import pathlib
import statistics
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import httpx
import pyarrow.parquet as pq
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pii import Tagger, combine, redact, regex_spans

HERE = pathlib.Path(__file__).resolve().parent
MODEL_DIR = os.environ.get("LOG_PII_MODEL_DIR", "/models/log-pii")
LOG_TABLE_URI = os.environ.get("LOG_PII_TABLE_URI", "")
TAIL_BATCH = 6
TAIL_PERIOD_S = 2.0
TAIL_KEEP = 60
MAX_LINE_CHARS = 500
SCANS_PER_MINUTE = 30
SQL_PER_MINUTE = 4

SQL_PRESETS = {
    "redact": """
SELECT ts, service,
       json_extract_string(r, '$.redacted') AS redacted,
       json_format(json_extract(r, '$.counts')) AS entities
FROM (
  SELECT ts, service, catalog_call('function.log-pii-redact', message) AS r
  FROM (SELECT ts, service, message FROM '@TABLE@' ORDER BY ts LIMIT 25)
)
ORDER BY ts""",
    "by_service": """
WITH scanned AS (
  SELECT service, catalog_call('function.log-pii-redact', message) AS r
  FROM '@TABLE@'
)
SELECT service,
       count(*) AS lines,
       sum(CASE WHEN json_array_length(json_extract(r, '$.spans')) > 0 THEN 1 ELSE 0 END) AS lines_with_pii,
       sum(json_array_length(json_extract(r, '$.spans'))) AS entities
FROM scanned
GROUP BY service
ORDER BY lines_with_pii DESC""",
}


class RateLimiter:
    """Per-client sliding window; clients are keyed by the first X-Forwarded-For hop."""

    def __init__(self, per_minute: int) -> None:
        super().__init__()
        self.per_minute = per_minute
        self.hits: dict[str, collections.deque[float]] = collections.defaultdict(collections.deque)
        self.lock = threading.Lock()

    def allow(self, request: Request) -> bool:
        key = (request.headers.get("x-forwarded-for") or (request.client.host if request.client else "?")).split(",")[0]
        now = time.monotonic()
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] > 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            if len(self.hits) > 10_000:
                self.hits.clear()
            return True


class ChalkSql:
    """Minimal ChalkSQL client over Connect JSON, with a cached client-credentials token."""

    def __init__(self) -> None:
        super().__init__()
        self.api = os.environ["CHALK_API_SERVER"].rstrip("/")
        self.env = os.environ["CHALK_ENVIRONMENT_ID"]
        self._client_id = os.environ["CHALK_CLIENT_ID"]
        self._secret = os.environ["CHALK_CLIENT_SECRET"]
        self._http = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))
        self._token, self._expiry = "", 0.0

    def _bearer(self) -> str:
        if time.time() > self._expiry - 300:
            r = self._http.post(
                f"{self.api}/v1/oauth/token",
                json={"client_id": self._client_id, "client_secret": self._secret, "grant_type": "client_credentials"},
            )
            r.raise_for_status()
            body = r.json()
            self._token, self._expiry = body["access_token"], time.time() + float(body.get("expires_in", 3600))
        return self._token

    def query(self, sql: str) -> dict[str, Any]:
        started = time.monotonic()
        r = self._http.post(
            f"{self.api}/chalk.protosql.v1.SqlService/ExecuteSqlQuery",
            headers={
                "Authorization": f"Bearer {self._bearer()}",
                "X-Chalk-Env-Id": self.env,
                "x-chalk-deployment-type": "engine-grpc",
                "Content-Type": "application/json",
            },
            json={"query": sql, "syncOptions": {}},
        )
        body = r.json()
        if "parquet" not in body:
            errors = body.get("errors") or [{"message": body.get("message") or json.dumps(body)[:300]}]
            raise RuntimeError(errors[0].get("message") or "query failed")
        table = pq.read_table(io.BytesIO(base64.b64decode(body["parquet"])))
        return {
            "columns": table.column_names,
            "rows": table.to_pylist(),
            "query_id": body.get("queryId"),
            "elapsed_ms": round((time.monotonic() - started) * 1000),
        }


class State:
    def __init__(self) -> None:
        super().__init__()
        self.tagger: Tagger | None = None
        self.infer_lock = threading.Lock()
        self.tail_lines = [json.loads(line)["text"] for line in (HERE / "data" / "tail.jsonl").open()]
        self.tail: collections.deque[dict] = collections.deque(maxlen=TAIL_KEEP)
        self.seq = 0
        self.batch_ms: collections.deque[float] = collections.deque(maxlen=200)
        self.scan_ms: collections.deque[float] = collections.deque(maxlen=200)
        self.metrics = json.loads((HERE / "metrics.json").read_text())
        self.sql_lock = asyncio.Lock()
        self.sql_cache: dict[str, tuple[float, dict]] = {}

    def scan(self, lines: list[str]) -> tuple[list[dict], float]:
        assert self.tagger is not None
        t0 = time.perf_counter()
        with self.infer_lock:
            model = self.tagger.spans(lines)
        ms = (time.perf_counter() - t0) * 1000
        out = []
        for line, m in zip(lines, model):
            regex = regex_spans(line)
            combined = combine(line, m, regex)
            out.append(
                {
                    "text": line,
                    "redacted": redact(line, combined),
                    "combined": [s.as_dict(line) for s in combined],
                    "model": [s.as_dict(line) for s in m],
                    "regex": [s.as_dict(line) for s in regex],
                }
            )
        return out, ms


state = State()
scan_limit = RateLimiter(SCANS_PER_MINUTE)
sql_limit = RateLimiter(SQL_PER_MINUTE)


async def tail_loop() -> None:
    cursor = 0
    while True:
        lines = [state.tail_lines[(cursor + i) % len(state.tail_lines)] for i in range(TAIL_BATCH)]
        cursor += TAIL_BATCH
        results, ms = await asyncio.to_thread(state.scan, lines)
        state.batch_ms.append(ms)
        for r in results:
            state.seq += 1
            state.tail.append({"seq": state.seq, "at": time.time(), **r})
        await asyncio.sleep(TAIL_PERIOD_S)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    torch.set_num_threads(int(os.environ.get("LOG_PII_THREADS", "2")))
    state.tagger = await asyncio.to_thread(Tagger, MODEL_DIR)
    await asyncio.to_thread(state.scan, state.tail_lines[:8])  # warm up
    task = asyncio.create_task(tail_loop())
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https://([a-z0-9-]+\.)*chalk\.(ai|dev)|https://[a-z0-9-]+\.vercel\.app|http://localhost:\d+",
    allow_methods=["GET", "POST"],
    allow_headers=["content-type"],
)


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": state.tagger is not None}


@app.get("/api/tail")
def tail(after: int = 0) -> dict:
    items = [t for t in state.tail if t["seq"] > after][-24:]
    return {"items": items, "seq": state.seq, "batch_size": TAIL_BATCH}


class ScanRequest(BaseModel):
    text: str


@app.post("/api/scan")
async def scan(req: ScanRequest, request: Request) -> dict:
    if not scan_limit.allow(request):
        raise HTTPException(429, "Too many scans; try again in a minute.")
    text = " ".join(req.text.split())[:MAX_LINE_CHARS]
    if not text:
        raise HTTPException(400, "Paste a log line to scan.")
    results, ms = await asyncio.to_thread(state.scan, [text])
    state.scan_ms.append(ms)
    return {**results[0], "model_ms": round(ms, 1), "truncated": len(" ".join(req.text.split())) > MAX_LINE_CHARS}


class SqlRequest(BaseModel):
    preset: str


@app.post("/api/sql")
async def sql(req: SqlRequest, request: Request) -> dict:
    if req.preset not in SQL_PRESETS or not LOG_TABLE_URI:
        raise HTTPException(400, "Unknown query.")
    cached = state.sql_cache.get(req.preset)
    if cached and time.time() - cached[0] < 20:
        return {**cached[1], "cached": True}
    if not sql_limit.allow(request):
        raise HTTPException(429, "Too many queries; try again in a minute.")
    query = SQL_PRESETS[req.preset].replace("@TABLE@", LOG_TABLE_URI)
    async with state.sql_lock:
        try:
            result = await asyncio.to_thread(ChalkSql().query, query)
        except Exception as exc:
            raise HTTPException(502, f"ChalkSQL query failed: {str(exc)[:200]}") from exc
    result = {**result, "sql": SQL_PRESETS[req.preset].strip().replace("'@TABLE@'", "app_logs")}
    state.sql_cache[req.preset] = (time.time(), result)
    return {**result, "cached": False}


def _pct(xs: collections.deque[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return round(s[min(len(s) - 1, int(q * len(s)))], 1)


@app.get("/api/metrics")
def metrics() -> dict:
    p50 = _pct(state.batch_ms, 0.5)
    return {
        **state.metrics,
        "live": {
            "tail_batch_size": TAIL_BATCH,
            "tail_batch_ms_p50": p50,
            "tail_batch_ms_p95": _pct(state.batch_ms, 0.95),
            "tail_lines_per_s": round(TAIL_BATCH / (p50 / 1000), 1) if p50 else None,
            "scan_ms_p50": _pct(state.scan_ms, 0.5),
            "scans_observed": len(state.scan_ms),
            "cpu": os.environ.get("LOG_PII_CPU", "2"),
            "stdev_ms": round(statistics.pstdev(state.batch_ms), 1) if len(state.batch_ms) > 1 else None,
        },
    }


if __name__ == "__main__":
    # access_log=False: request lines (and anything a client might put in a URL) are never written.
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), access_log=False)
