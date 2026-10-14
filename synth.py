#!/usr/bin/env -S uv run --script
#
# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = []
# ///
"""Synthetic application logs with labelled PII spans.

Every value is fictional or comes from a reserved range: phones in 555-0100..555-0199, emails on
example.com / example.org / *.test, test card numbers, SSNs in the 987-65-4320..987-65-4329
advertising range, IPs in the RFC 5737 / RFC 3849 documentation blocks, and made-up names and
streets. Decoys look like PII but are not (order IDs shaped like phone numbers, tracking numbers
that fail Luhn, four-part version strings), so a detector has to tell them apart.

The train and test splits draw from disjoint template families and disjoint name, street and city
pools, so test scores measure generalization rather than memorized templates.

    ./synth.py --out data            # writes data/{train,test}.jsonl and data/tail.jsonl
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import string
from dataclasses import dataclass
from typing import Callable

LABELS = ["NAME", "EMAIL", "PHONE", "ADDRESS", "CARD", "SSN", "IP", "API_KEY"]

FIRST_TRAIN = [
    "Ada",
    "Basil",
    "Cora",
    "Dmitri",
    "Elena",
    "Felix",
    "Greta",
    "Hugo",
    "Iris",
    "Jonah",
    "Kira",
    "Leo",
    "Mira",
    "Nils",
    "Opal",
    "Pavel",
    "Quinn",
    "Rosa",
    "Silas",
    "Tess",
    "Uma",
    "Viktor",
    "Wren",
    "Yara",
    "Zane",
    "Amara",
    "Bram",
    "Clio",
    "Dario",
    "Esme",
]
LAST_TRAIN = [
    "Quillfeather",
    "Brightwater",
    "Ashgrove",
    "Thornbury",
    "Mossbank",
    "Halloway",
    "Fernsby",
    "Wexley",
    "Pemberton-Vale",
    "Starling",
    "Okonkwo-Reyes",
    "Lindqvist",
    "Marchetti",
    "Castellanos",
    "Nakamura-Holt",
    "Whitlock",
    "Redfern",
    "Calloway",
]
FIRST_TEST = [
    "Bianca",
    "Cassius",
    "Delphine",
    "Emeric",
    "Fiora",
    "Gideon",
    "Hallie",
    "Ignatius",
    "Juniper",
    "Kasimir",
    "Lucinda",
    "Magnus",
    "Noor",
    "Orrin",
    "Priya",
    "Rafferty",
]
LAST_TEST = [
    "Wintergreen",
    "Duskmoor",
    "Larkspur",
    "Oakenshield",
    "Vanterpool",
    "Hargreave",
    "Esterhazy",
    "Blackwood-Ito",
    "Merriweather",
    "Szabo",
    "Delacroix",
    "Penhaligon",
]

STREET_TRAIN = ["Maple Hollow", "Cedar Ridge", "Lantern", "Willow Bend", "Copperfield", "Juniper"]
STREET_TEST = ["Foxglove", "Harbor Light", "Quarry", "Bramblewood", "Saltmarsh"]
SUFFIX = ["Rd", "Street", "Ave", "Lane", "Blvd", "Ct", "Way"]
CITY_TRAIN = [("Fernvale", "OR"), ("Lake Corvin", "MN"), ("Port Ellery", "ME"), ("Ashby Falls", "CO")]
CITY_TEST = [("Glenmarrow", "VT"), ("Redstone Mills", "NC"), ("Kettle Bay", "WA")]

EMAIL_DOMAINS = ["example.com", "example.org", "mail.example.net", "shop.test", "corp.test"]
TEST_CARDS = [
    "4111 1111 1111 1111",
    "4111-1111-1111-1111",
    "5555555555554444",
    "4012888888881881",
    "378282246310005",
    "6011111111111117",
    "5105 1051 0510 5100",
    "4000056655665556",
]


@dataclass
class Pools:
    first: list[str]
    last: list[str]
    street: list[str]
    city: list[tuple[str, str]]


def _name(r: random.Random, p: Pools) -> str:
    return f"{r.choice(p.first)} {r.choice(p.last)}"


def _email(r: random.Random, p: Pools) -> str:
    first, last = r.choice(p.first).lower(), r.choice(p.last).lower().replace("-", "")
    local = r.choice([f"{first}.{last}", f"{first[0]}{last}", f"{first}{r.randint(1, 99)}", f"{last}+orders"])
    return f"{local}@{r.choice(EMAIL_DOMAINS)}"


def _phone(r: random.Random, _p: Pools) -> str:
    area, line = r.choice(["212", "415", "312", "617", "503", "720"]), f"01{r.randint(0, 99):02d}"
    return r.choice([f"({area}) 555-{line}", f"{area}-555-{line}", f"+1 {area} 555 {line}", f"{area}.555.{line}"])


def _address(r: random.Random, p: Pools) -> str:
    city, state = r.choice(p.city)
    return f"{r.randint(10, 9899)} {r.choice(p.street)} {r.choice(SUFFIX)}, {city}, {state} {r.randint(10000, 99899)}"


def _card(r: random.Random, _p: Pools) -> str:
    return r.choice(TEST_CARDS)


def _ssn(r: random.Random, _p: Pools) -> str:
    return f"987-65-432{r.randint(0, 9)}"


def _ip(r: random.Random, _p: Pools) -> str:
    return r.choice(
        [
            f"192.0.2.{r.randint(1, 254)}",
            f"198.51.100.{r.randint(1, 254)}",
            f"203.0.113.{r.randint(1, 254)}",
            f"2001:db8::{r.randint(1, 0xFFFF):x}",
        ]
    )


def _api_key(r: random.Random, _p: Pools) -> str:
    body = "".join(r.choices(string.ascii_letters + string.digits, k=r.choice([24, 32])))
    return r.choice(["sk_test_", "pk_test_", "ghp_EXAMPLE", "xoxb-demo-", "AKIAEXAMPLE"]) + body


# Decoys: values a naive detector flags. They are never labelled.
def _order_id(r: random.Random, _p: Pools) -> str:
    # 10 digits with a leading 0 or 1 cannot be a NANP number (area codes start 2-9).
    return f"{r.choice('01')}{r.randint(100000000, 999999999)}"


def _tracking(r: random.Random, _p: Pools) -> str:
    while True:
        n = "".join(r.choices(string.digits, k=16))
        if not luhn_ok(n):
            return n


def _version(r: random.Random, _p: Pools) -> str:
    return f"{r.randint(1, 9)}.{r.randint(0, 30)}.{r.randint(0, 99)}.{r.randint(0, 9999)}"


def _hexid(r: random.Random, _p: Pools) -> str:
    return "".join(r.choices("0123456789abcdef", k=r.choice([12, 16, 32])))


def luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 13:
        return False
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


FILLERS: dict[str, Callable[[random.Random, Pools], str]] = {
    "NAME": _name,
    "EMAIL": _email,
    "PHONE": _phone,
    "ADDRESS": _address,
    "CARD": _card,
    "SSN": _ssn,
    "IP": _ip,
    "API_KEY": _api_key,
    "order": _order_id,
    "tracking": _tracking,
    "version": _version,
    "hex": _hexid,
}

# {X} is labelled PII; {x} (lowercase) is an unlabelled decoy.
TRAIN_TEMPLATES = [
    "INFO checkout: user {NAME} <{EMAIL}> placed order {order}",
    "WARN payments: card {CARD} declined for {NAME}, retrying (txn {hex})",
    "ERROR notify: SMS to {PHONE} failed: carrier timeout",
    '{{"level":"info","svc":"signup","email":"{EMAIL}","ip":"{IP}","msg":"account created"}}',
    '{{"level":"debug","svc":"kyc","ssn":"{SSN}","name":"{NAME}","status":"pending_review"}}',
    "DEBUG shipping: label printed for {NAME}, {ADDRESS}; tracking {tracking}",
    '{IP} - - [14/Oct/2026:10:22:31 +0000] "POST /api/v2/login HTTP/1.1" 401 512',
    "WARN auth: rejected request with Authorization: Bearer {API_KEY}",
    "INFO support: ticket opened by {NAME} ({PHONE}) about order {order}",
    "ERROR export: failed to geocode '{ADDRESS}' for customer {hex}",
    "INFO deploy: agent {version} connected from {IP}",
    "DEBUG crm: merged duplicate contacts {EMAIL} and {EMAIL}",
    "WARN billing: refund issued to {NAME}, card ending {CARD}",
    '{{"event":"password_reset","to":"{EMAIL}","phone":"{PHONE}","requested_by":"{IP}"}}',
    "INFO worker: job {hex} finished in 412ms (build {version})",
    "ERROR webhook: config contains a live-looking key {API_KEY}; rotating",
    "INFO orders: order {order} for {NAME} shipped to {ADDRESS}",
    "DEBUG tax: verifying SSN {SSN} for {NAME}",
    "INFO cdn: cache miss for tracking page {tracking} from {IP}",
    "WARN fraud: velocity check tripped for {EMAIL} on card {CARD}",
]
TEST_TEMPLATES = [
    "NOTICE mailer: bounced message to {EMAIL} (recipient {NAME})",
    '{{"lvl":"warn","component":"courier","dest":"{ADDRESS}","contact":"{PHONE}","parcel":"{tracking}"}}',
    "ERROR ledger: chargeback on {CARD} disputed by {NAME}; case {hex}",
    "INFO gateway: {IP} called /v1/charges with key {API_KEY}",
    "DEBUG identity: SSN {SSN} matched applicant {NAME} at {ADDRESS}",
    "INFO fulfilment: reminder call scheduled for {PHONE} re: order {order}",
    "WARN session: login for {EMAIL} from new device {IP}",
    "INFO updater: rolled out client {version} to 3 regions",
    "ERROR crm-sync: could not reach {NAME} at {PHONE}; falling back to {EMAIL}",
    '{{"msg":"invoice sent","customer":"{NAME}","bill_to":"{ADDRESS}","ref":"{order}"}}',
]

_FIELD = string.Formatter()


def render(template: str, r: random.Random, pools: Pools) -> dict:
    text, spans = "", []
    for literal, field, _, _ in _FIELD.parse(template):
        text += literal
        if field is None:
            continue
        value = FILLERS[field](r, pools)
        if field in LABELS:
            spans.append({"start": len(text), "end": len(text) + len(value), "label": field})
        text += value
    ts = f"2026-10-14T{r.randint(0, 23):02d}:{r.randint(0, 59):02d}:{r.randint(0, 59):02d}Z "
    shifted = [{**s, "start": s["start"] + len(ts), "end": s["end"] + len(ts)} for s in spans]
    return {"text": ts + text, "spans": shifted}


def generate(templates: list[str], pools: Pools, n: int, seed: int) -> list[dict]:
    r = random.Random(seed)
    return [render(r.choice(templates), r, pools) for _ in range(n)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--train", type=int, default=24000)
    parser.add_argument("--test", type=int, default=2000)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    train_pools = Pools(FIRST_TRAIN, LAST_TRAIN, STREET_TRAIN, CITY_TRAIN)
    test_pools = Pools(FIRST_TEST, LAST_TEST, STREET_TEST, CITY_TEST)
    splits = {
        "train": generate(TRAIN_TEMPLATES, train_pools, args.train, 1),
        "test": generate(TEST_TEMPLATES, test_pools, args.test, 2),
        # The page's live tail mixes both template families, always with held-out names.
        "tail": generate(TRAIN_TEMPLATES + TEST_TEMPLATES, test_pools, 600, 3),
    }
    for name, rows in splits.items():
        with (args.out / f"{name}.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        print(f"{name}: {len(rows)} lines")


if __name__ == "__main__":
    main()
