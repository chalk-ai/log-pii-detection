#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["chalkcompute==2.13.5"]
# ///
"""Deploy the /demos/log-pii-detection backend (server.py) as a ScalingGroup.

The app runs the same tagger in-process for the live tail and the paste box, and calls ChalkSQL
(which calls `function.log-pii-redact`) for the log-table queries. Chalk credentials are passed as
Chalk secrets, never baked into the image:

    ./deploy.py --model-dir model/ --table-uri gs://<bucket>/demos/log-pii-detection/app_logs.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess

HERE = pathlib.Path(__file__).resolve().parent
MODEL_DIR = "/models/log-pii"


def load_chalk_env(env_id: str | None) -> str:
    cfg = json.loads(subprocess.check_output(["chalk", "config", "--json"]))
    os.environ.setdefault("CHALK_API_SERVER", cfg["apiServer"]["value"])
    os.environ.setdefault("CHALK_CLIENT_ID", cfg["clientId"]["value"])
    os.environ.setdefault("CHALK_CLIENT_SECRET", cfg["clientSecret"]["value"])
    os.environ["CHALK_ENVIRONMENT_ID"] = env_id or cfg["environment"]["value"]
    # Distinct local names, so the upserted Chalk secrets do not collide with other workloads.
    os.environ["LOG_PII_CLIENT_ID"] = os.environ["CHALK_CLIENT_ID"]
    os.environ["LOG_PII_CLIENT_SECRET"] = os.environ["CHALK_CLIENT_SECRET"]
    return os.environ["CHALK_ENVIRONMENT_ID"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="log-pii-detection")
    parser.add_argument("--env-id", default=None)
    parser.add_argument("--model-dir", type=pathlib.Path, required=True)
    parser.add_argument("--table-uri", required=True)
    parser.add_argument("--ready-timeout", type=float, default=1200.0)
    args = parser.parse_args()

    env_id = load_chalk_env(args.env_id)
    from chalkcompute import Image, ReadinessProbe, ScalingGroup, Secret

    image = (
        Image.debian_slim(python_version="3.12")
        .run_commands("pip install --no-cache-dir torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu")
        .pip_install(
            [
                "transformers==4.56.2",
                "fastapi==0.116.1",
                "uvicorn[standard]==0.35.0",
                "httpx==0.28.1",
                "pyarrow==21.0.0",
            ]
        )
        .add_local_dir(str(args.model_dir), MODEL_DIR, strategy="volume")
        .add_local_dir(
            str(HERE),
            "/app",
            strategy="volume",
            exclude=["model/**", "data/train.jsonl", "data/test.jsonl", "__pycache__/**", "*.pyc"],
        )
        .workdir("/app")
    )
    sg = ScalingGroup(
        image=image,
        name=args.name,
        cpu="2",
        memory="4Gi",
        env={
            "CHALK_API_SERVER": os.environ["CHALK_API_SERVER"],
            "CHALK_ENVIRONMENT_ID": env_id,
            "LOG_PII_MODEL_DIR": MODEL_DIR,
            "LOG_PII_TABLE_URI": args.table_uri,
            "LOG_PII_THREADS": "2",
            "LOG_PII_CPU": "2",
            "MALLOC_ARENA_MAX": "2",
            "PORT": "8080",
        },
        secrets=[
            Secret.from_local_env("LOG_PII_CLIENT_ID", env_var_name="CHALK_CLIENT_ID"),
            Secret.from_local_env("LOG_PII_CLIENT_SECRET", env_var_name="CHALK_CLIENT_SECRET"),
        ],
        entrypoint=["python", "/app/server.py"],
        port=8080,
        readiness_probe=ReadinessProbe.http("/healthz"),
        min_replicas=1,
        max_replicas=1,
    )
    sg.deploy(build_timeout=1800.0, ready_timeout=args.ready_timeout)
    print("web_url:", sg.web_url)


if __name__ == "__main__":
    main()
