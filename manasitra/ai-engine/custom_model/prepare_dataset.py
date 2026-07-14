"""
Manasitra Dataset Preparation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Merges 3 sources into one clean therapy_dataset.json:

1. train.csv       — 3,512 real therapist Q&A pairs (Kaggle)
2. KB.json         — 80 intents, ~5,900 pattern-response pairs
3. therapy_dataset.json — our hand-crafted 91 pairs

Output: therapy_dataset.json (merged, cleaned, deduplicated)
"""

import csv
import json
import os
import re
import random

ROOT         = os.path.join(os.path.dirname(__file__), "../../..")
TRAIN_CSV    = os.path.join(ROOT, "train.csv")
KB_JSON      = os.path.join(ROOT, "KB.json")
EXISTING     = os.path.join(os.path.dirname(__file__), "psychology_dataset.json")
OUTPUT       = os.path.join(os.path.dirname(__file__), "psychology_dataset.json")

MAX_INPUT_LEN    = 200   # chars — truncate very long inputs
MAX_RESPONSE_LEN = 400   # chars — truncate very long responses
MIN_RESPONSE_LEN = 30    # skip too-short responses


def clean(text: str) -> str:
    """Clean and normalize text."""
    text = text.strip()
    # Remove excessive whitespace / newlines
    text = re.sub(r'\s+', ' ', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove phone numbers (keep crisis numbers handled separately)
    text = re.sub(r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b', '', text)
    return text.strip()


def truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    # Cut at last sentence boundary before max_len
    cut = text[:max_len]
    last_period = max(cut.rfind('.'), cut.rfind('!'), cut.rfind('?'))
    if last_period > max_len // 2:
        return cut[:last_period + 1]
    return cut + "..."


pairs: list[dict] = []
seen: set[str] = set()


def add(inp: str, resp: str):
    inp  = truncate(clean(inp),  MAX_INPUT_LEN)
    resp = truncate(clean(resp), MAX_RESPONSE_LEN)
    if len(resp) < MIN_RESPONSE_LEN:
        return
    key = inp[:60].lower()
    if key in seen:
        return
    seen.add(key)
    pairs.append({"input": inp, "response": resp})


# ── 1. Load existing hand-crafted dataset ────────────────────────────────────
print("Loading existing therapy_dataset.json...")
if os.path.exists(EXISTING):
    with open(EXISTING, encoding="utf-8") as f:
        existing = json.load(f)
    for item in existing:
        add(item["input"], item["response"])
    print(f"  Loaded {len(pairs)} existing pairs")


# ── 2. Load train.csv (real therapist conversations) ─────────────────────────
print("Skipping train.csv (file deleted)...")


# ── 3. Load KB.json (intent patterns + responses) ────────────────────────────
print("Loading KB.json...")
kb_count = 0
with open(KB_JSON, encoding="utf-8") as f:
    kb = json.load(f)

for intent in kb["intents"]:
    patterns  = intent.get("patterns", [])
    responses = intent.get("responses", intent.get("response", []))
    if isinstance(responses, str):
        responses = [responses]

    # Skip empty/trivial patterns
    patterns = [p for p in patterns if len(p.strip()) > 3]
    if not patterns or not responses:
        continue

    # Pair each pattern with a random response (avoid combinatorial explosion)
    for pattern in patterns:
        resp = random.choice(responses)
        add(pattern, resp)
        kb_count += 1

print(f"  Loaded {kb_count} pairs from KB.json")


# ── 4. Shuffle and save ───────────────────────────────────────────────────────
random.shuffle(pairs)
print(f"\nTotal unique pairs: {len(pairs)}")

with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(pairs, f, indent=2, ensure_ascii=False)

print(f"Saved to: {OUTPUT}")
print("\nSample pairs:")
for p in pairs[:3]:
    print(f"  INPUT   : {p['input'][:80]}")
    print(f"  RESPONSE: {p['response'][:80]}")
    print()
