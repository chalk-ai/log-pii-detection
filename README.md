# PII detection in logs with a fine-tuned DistilBERT

The code behind [chalk.ai/demos/log-pii-detection](https://chalk.ai/demos/log-pii-detection). A token classifier fine-tuned from
`distilbert/distilbert-base-cased` (Apache-2.0) tags eight PII types in log lines: names, emails,
phones, street addresses, card numbers, SSNs, IPs and API keys. Cheap validators keep structured
values honest (Luhn for cards, NANP shape for phones, valid IPs outside version strings, SSN group
rules), so look-alikes such as order IDs, tracking numbers and build versions are not redacted.

All data is synthetic: phones in 555-0100..0199, example.* and .test email domains, published test
card numbers, the 987-65-4320..4329 advertising SSNs, RFC 5737 / RFC 3849 documentation IPs, and
made-up names and streets.

| File | Role |
| --- | --- |
| `synth.py` | Labelled synthetic logs. Train and test use disjoint templates and name/street/city pools. |
| `pii.py` | Regex detectors, validators, BIO decoding, `combine`, `redact`, and the `Tagger`. |
| `train.py` | Fine-tunes DistilBERT on CPU (about 30 minutes on 48 cores for 2 epochs). |
| `evaluate.py` | Precision/recall for regex, model and combined on the held-out split → `metrics.json`. |
| `deploy_function.py` | Deploys `function.log-pii-redact`, callable from ChalkSQL with `catalog_call`. |
| `make_table.py` | Writes the demo's `app_logs` parquet table. |
| `server.py`, `deploy.py` | The page's API (live tail, paste scan, ChalkSQL presets) as a ScalingGroup. |

```bash
./synth.py --out data
./train.py --data data --out model
./evaluate.py --data data --model model --out metrics.json
./deploy_function.py --model-dir model
./make_table.py --data data --out data/app_logs.parquet
gcloud storage cp data/app_logs.parquet gs://<bucket>/demos/log-pii-detection/
./deploy.py --model-dir model --table-uri gs://<bucket>/demos/log-pii-detection/app_logs.parquet
```

```sql
SELECT ts, service, json_extract_string(r, '$.redacted') AS redacted
FROM (SELECT ts, service, catalog_call('function.log-pii-redact', message) AS r FROM app_logs)
```

## License

Apache-2.0. The base model, `distilbert/distilbert-base-cased`, is Apache-2.0; the fine-tuned weights are trained only on the synthetic data `synth.py` generates.

## Local authentication

The SDK reads credentials, API host, and the selected environment from your existing
`chalk login` in `~/.chalk.yml`, including project-scoped and default logins.
For unattended runs, `CHALK_CLIENT_ID` and `CHALK_CLIENT_SECRET` remain supported.
The examples use the current environment selected by your Chalk login.
