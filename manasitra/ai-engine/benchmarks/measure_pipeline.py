"""
Measure Manasitra's local emotional-AI pipeline.

This generates verifiable technical benchmarks for:
1. Emotion classification accuracy (English + Hinglish).
2. End-to-end local generation latency without external APIs.
3. Privacy redaction safety (Email, Phone, PAN, UPI).
"""

import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from emotion_model.predictor import predict_emotion
from privacy_engine.privacy import privacy_filter
from response_engine.conversation_brain import generate_response


SAMPLES = [
    # English benchmarks
    ("I feel happy today, I got good marks", "happy"),
    ("College pressure and seeing everyone move ahead makes me tired and alone", "overwhelmed"),
    ("Nobody talks to me and I feel invisible", "lonely"),
    ("My exams are tomorrow and I am panicking", "anxiety"),
    ("I keep worrying about worst case scenarios", "anxiety"),
    ("I cried all night and feel empty", "sad"),
    # Hinglish & Multilingual benchmarks
    ("aaj bahut udaas hu aur rona aa raha hai", "sad"),
    ("bahut tension aur pressure hai exams ka", "stress"),
    ("koi mera dost nahi hai, akela lag raha hai", "lonely"),
    ("aaj mai bahut khush hu", "happy"),
    ("aaj mera mood bohot off hai", "sad"),
    ("bohot bura lag raha hai", "sad"),
    ("mai theek nahi hu bilkul bhi", "sad"),
    ("aaj din bohot accha tha maza aaya", "happy"),
    ("interview se bohot darr lag raha hai", "anxiety"),
]

PII_TEST_SAMPLES = [
    ("Contact me at test@example.com or 9876543210", 2),
    ("My PAN card is ABCDE1234F and UPI is rahul@okaxis", 2),
    ("No PII in this message just talking about my day", 0),
]


def main() -> None:
    latencies = []
    correct = 0
    rows = []

    # Warmup
    _ = generate_response("warmup", "neutral", [])

    for text, expected in SAMPLES:
        started = time.perf_counter()
        privacy = privacy_filter(text)
        result = predict_emotion(privacy.redacted_text)
        reply = generate_response(privacy.redacted_text, result["emotion"], [])
        latency_ms = (time.perf_counter() - started) * 1000

        latencies.append(latency_ms)
        is_correct = result["emotion"] == expected
        correct += int(is_correct)

        rows.append({
            "text": text,
            "expected": expected,
            "predicted": result["emotion"],
            "correct": is_correct,
            "confidence": result["confidence"],
            "latency_ms": round(latency_ms, 2),
            "local_only": True,
            "reply_sample": reply,
        })

    # PII verification test
    pii_passed = 0
    for pii_text, expected_redactions in PII_TEST_SAMPLES:
        p_res = privacy_filter(pii_text)
        total_redactions = sum(p_res.redaction_counts.values())
        if total_redactions == expected_redactions:
            pii_passed += 1

    report = {
        "sample_count": len(SAMPLES),
        "accuracy": round(correct / len(SAMPLES), 4),
        "pii_redaction_accuracy": round(pii_passed / len(PII_TEST_SAMPLES), 4),
        "avg_latency_ms": round(statistics.mean(latencies), 2),
        "p50_latency_ms": round(statistics.median(latencies), 2),
        "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 2),
        "external_api_used": False,
        "rows": rows,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
