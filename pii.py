"""PII detection for log lines: a token-classification model, regex detectors, and validators.

`detect(lines, mode)` returns character spans per line for one of three detectors:

- ``regex``: pattern matching only (no validators), the usual baseline.
- ``model``: the fine-tuned BERT-family tagger only.
- ``combined``: what the demo ships. Free-text entities (names, addresses) come from the model;
  structured entities (emails, phones, cards, SSNs, IPs, API keys) are accepted from the model or
  regex only when their validator passes, which removes look-alikes such as order IDs shaped like
  phone numbers or tracking numbers that fail the Luhn check.
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Literal, Sequence

LABELS = ["NAME", "EMAIL", "PHONE", "ADDRESS", "CARD", "SSN", "IP", "API_KEY"]
TAGS = ["O"] + [f"{p}-{label}" for label in LABELS for p in ("B", "I")]
STRUCTURED = {"EMAIL", "PHONE", "CARD", "SSN", "IP", "API_KEY"}
Mode = Literal["regex", "model", "combined"]


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    source: str  # "model" or "regex"

    def as_dict(self, text: str) -> dict:
        return {
            "start": self.start,
            "end": self.end,
            "label": self.label,
            "source": self.source,
            "text": text[self.start : self.end],
        }


REGEXES: dict[str, re.Pattern[str]] = {
    "EMAIL": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    "PHONE": re.compile(r"(?<![\w.])(?:\+1[ .-]?)?(?:\(\d{3}\)\s?|\d{3}[ .-]?)\d{3}[ .-]?\d{4}(?![\w])"),
    "CARD": re.compile(r"(?<![\w-])(?:\d[ -]?){12,18}\d(?![\w-])"),
    "SSN": re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])"),
    "IP": re.compile(r"(?<![\w.:])(?:(?:\d{1,3}\.){3}\d{1,3}|[0-9a-fA-F]{1,4}(?::[0-9a-fA-F]{0,4}){2,7})(?![\w.:])"),
    "API_KEY": re.compile(
        r"(?<![\w-])(?:(?:sk|pk)_(?:test|live)_[A-Za-z0-9]{16,}|ghp_[A-Za-z0-9]{20,}"
        r"|xox[bp]-[A-Za-z0-9-]{10,}|AKIA[A-Z0-9]{12,}[A-Za-z0-9]*)"
    ),
    "ADDRESS": re.compile(
        r"\b\d{1,5} [A-Z][a-z]+(?: [A-Z][a-z]+)* (?:St|Street|Rd|Road|Ave|Avenue|Ln|Lane"
        r"|Blvd|Ct|Court|Way|Dr|Drive)\b(?:, [A-Z][a-z]+(?: [A-Z][a-z]+)*, [A-Z]{2} \d{5})?"
    ),
}

_VERSION_CONTEXT = re.compile(r"(?:\bv|version|build|client|agent|release)\W{0,3}$", re.I)


def luhn_ok(digits: str) -> bool:
    nums = [int(c) for c in digits if c.isdigit()]
    if not 13 <= len(nums) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def valid(label: str, text: str, start: int, end: int) -> bool:
    """Cheap structural checks that a real value of this type must pass."""
    value = text[start:end]
    if label == "EMAIL":
        return REGEXES["EMAIL"].fullmatch(value) is not None
    if label == "PHONE":
        digits = re.sub(r"\D", "", value)
        if len(digits) == 11 and digits[0] == "1":
            digits = digits[1:]
        # NANP: area code and exchange both start with 2-9.
        return len(digits) == 10 and digits[0] in "23456789" and digits[3] in "23456789"
    if label == "CARD":
        return luhn_ok(value)
    if label == "SSN":
        m = re.fullmatch(r"(\d{3})-(\d{2})-(\d{4})", value)
        return bool(m) and m.group(1) not in ("000", "666") and m.group(2) != "00" and m.group(3) != "0000"
    if label == "IP":
        try:
            ipaddress.ip_address(value)
        except ValueError:
            return False
        # A dotted quad right after "build", "client", "v" and similar is a version string.
        return _VERSION_CONTEXT.search(text[max(0, start - 12) : start]) is None
    if label == "API_KEY":
        return len(value) >= 20 and any(c.isdigit() for c in value) and any(c.isalpha() for c in value)
    return True


def regex_spans(text: str) -> list[Span]:
    spans = [Span(m.start(), m.end(), label, "regex") for label, rx in REGEXES.items() for m in rx.finditer(text)]
    return _dedupe(spans)


def _dedupe(spans: Sequence[Span]) -> list[Span]:
    """Drop spans that overlap a longer kept span (e.g. an SSN that is also inside a card match)."""
    kept: list[Span] = []
    for s in sorted(spans, key=lambda s: (-(s.end - s.start), s.start)):
        if all(s.end <= k.start or s.start >= k.end for k in kept):
            kept.append(s)
    return sorted(kept, key=lambda s: s.start)


def decode_tags(text: str, offsets: Sequence[tuple[int, int]], tag_ids: Sequence[int]) -> list[Span]:
    """Turn per-token BIO tags into character spans; subword pieces extend the current span."""
    spans: list[Span] = []
    cur: list | None = None
    for (s, e), tid in zip(offsets, tag_ids):
        if s == e:
            continue
        tag = TAGS[tid]
        if tag == "O":
            if cur:
                spans.append(Span(cur[0], cur[1], cur[2], "model"))
                cur = None
            continue
        prefix, label = tag.split("-", 1)
        if cur and cur[2] == label and (prefix == "I" or s == cur[1]):
            cur[1] = e
        else:
            if cur:
                spans.append(Span(cur[0], cur[1], cur[2], "model"))
            cur = [s, e, label]
    if cur:
        spans.append(Span(cur[0], cur[1], cur[2], "model"))
    return [_trim(text, s) for s in spans]


def _trim(text: str, s: Span) -> Span:
    start, end = s.start, s.end
    # A span that stops inside a word lost its last word piece to an O tag; take the whole word.
    while end < len(text) and text[end].isalnum() and text[end - 1].isalnum():
        end += 1
    while start > 0 and text[start - 1].isalnum() and text[start].isalnum():
        start -= 1
    while end > start and text[end - 1] in ".,;:'\")]}":
        end -= 1
    while start < end and text[start] in "'\"([{":
        start += 1
    return Span(start, end, s.label, s.source)


def combine(text: str, model: Sequence[Span], regex: Sequence[Span]) -> list[Span]:
    picked = [s for s in model if s.label not in STRUCTURED or valid(s.label, text, s.start, s.end)]
    picked += [s for s in regex if s.label in STRUCTURED and valid(s.label, text, s.start, s.end)]
    return _dedupe(picked)


def redact(text: str, spans: Sequence[Span]) -> str:
    out, pos = [], 0
    for s in sorted(spans, key=lambda s: s.start):
        out.append(text[pos : s.start])
        out.append(f"[{s.label}]")
        pos = s.end
    out.append(text[pos:])
    return "".join(out)


class Tagger:
    """The fine-tuned token classifier; imported lazily so regex-only callers need no torch."""

    def __init__(self, model_dir: str, max_length: int = 160) -> None:
        super().__init__()
        import torch
        from transformers import AutoModelForTokenClassification, AutoTokenizer

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(model_dir).eval()
        self.max_length = max_length

    def spans(self, lines: Sequence[str]) -> list[list[Span]]:
        if not lines:
            return []
        enc = self.tokenizer(
            list(lines),
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping").tolist()
        with self._torch.inference_mode():
            tag_ids = self.model(**enc).logits.argmax(-1).tolist()
        mask = enc["attention_mask"].tolist()
        return [
            decode_tags(line, [tuple(o) for o, m in zip(offs, msk) if m], [t for t, m in zip(tids, msk) if m])
            for line, offs, tids, msk in zip(lines, offsets, tag_ids, mask)
        ]


def detect(lines: Sequence[str], mode: Mode, tagger: Tagger | None) -> list[list[Span]]:
    if mode == "regex":
        return [regex_spans(line) for line in lines]
    assert tagger is not None
    model = tagger.spans(lines)
    if mode == "model":
        return model
    return [combine(line, m, regex_spans(line)) for line, m in zip(lines, model)]
