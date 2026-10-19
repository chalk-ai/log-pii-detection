#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Deploy the PII detector as a Chalk function, callable from any ChalkSQL query:

./deploy_function.py --model-dir model/

SELECT catalog_call('function.log-pii-redact', message) FROM app_logs
"""

from __future__ import annotations

import argparse
import json
import os
import threading
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = "/models/log-pii"
PII_DIR = "/opt/log_pii"
_tagger: list = []
_load_lock = threading.Lock()
_infer_lock = threading.Lock()


def log_pii_redact(line: str) -> str:
    """Tag PII in one log line; returns JSON with the redacted line, spans and per-entity counts."""
    import sys

    # The function build owns /app (handler.py, user_module.py), so pii.py ships under /opt.
    if PII_DIR not in sys.path:
        sys.path.insert(0, PII_DIR)
    from pii import Tagger, detect, redact

    if not _tagger:
        with _load_lock:
            if not _tagger:
                _tagger.append(Tagger(MODEL_DIR))
    text = line or ""
    with _infer_lock:
        spans = detect([text], "combined", _tagger[0])[0]
    return json.dumps(
        {
            "redacted": redact(text, spans),
            "spans": [{"start": s.start, "end": s.end, "label": s.label, "source": s.source} for s in spans],
            "counts": dict(Counter(s.label for s in spans)),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--model-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.env_id:
        os.environ["CHALK_ENVIRONMENT"] = args.env_id
    import chalkcompute
    from chalkcompute import Image, RemoteFunction

    image = (
        Image.debian_slim(python_version="3.12")
        .run_commands("pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu")
        .pip_install(["transformers==4.56.2"])
        .add_local_file(str(HERE / "pii.py"), f"{PII_DIR}/pii.py", strategy="copy")
        .add_local_dir(str(args.model_dir), MODEL_DIR, strategy="volume")
    )
    name = "log-pii-redact"
    try:
        chalkcompute.function(
            image=image,
            name=name,
            cpu="2",
            memory="4Gi",
            env={"MALLOC_ARENA_MAX": "2", "OMP_NUM_THREADS": "2"},
            min_replicas=1,
            max_replicas=2,
            max_batching_size=64,
        )(log_pii_redact)
    except Exception as exc:
        if "not ready" not in str(exc):
            raise
        print(f"{name}: {exc}; waiting longer")
        RemoteFunction.from_name(name).wait_ready(timeout=1200)
    print(f"deployed function.{name}")


if __name__ == "__main__":
    main()
