#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["pyarrow==21.0.0"]
# ///
"""Write the demo's `app_logs` table (ts, service, level, message) from synthetic log lines.

./make_table.py --data data --out data/app_logs.parquet
gcloud storage cp data/app_logs.parquet gs://<bucket>/demos/log-pii-detection/
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
from datetime import datetime

import pyarrow as pa
import pyarrow.parquet as pq

LINE = re.compile(r"^(\S+Z) (?:(INFO|WARN|ERROR|DEBUG|NOTICE) ([\w-]+): )?(.*)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data/app_logs.parquet"))
    parser.add_argument("--rows", type=int, default=240)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.data / "tail.jsonl").open()][: args.rows]
    cols: dict[str, list] = {"ts": [], "service": [], "level": [], "message": []}
    for r in rows:
        m = LINE.match(r["text"])
        assert m
        msg = m.group(4)
        svc = m.group(3)
        if svc is None:
            j = re.search(r'"(?:svc|component)":"([\w-]+)"', msg)
            svc = j.group(1) if j else ("edge" if msg[:1].isdigit() else "app")
        cols["ts"].append(datetime.fromisoformat(m.group(1).replace("Z", "+00:00")))
        cols["service"].append(svc)
        cols["level"].append(m.group(2) or "INFO")
        cols["message"].append(r["text"])
    pq.write_table(pa.table(cols), args.out)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
