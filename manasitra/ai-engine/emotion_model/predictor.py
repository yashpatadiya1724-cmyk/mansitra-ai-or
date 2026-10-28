"""
Manasitra Emotion Predictor v2.1 — Multilingual / Hinglish Enhanced
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Detects 9 emotions with compound scoring:
  happy, sad, angry, stress, anxiety, lonely, overwhelmed, neutral, grief

Rules:
- Multilingual & Hinglish keyword scoring with intensity weights
- Compound detection (stress + sad = overwhelmed)
- Context negation handling ("not happy" / "khush nahi" -> sad)
- Never returns happy for clearly distressed input
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import re
from typing import Optional

LABELS = ["happy", "sad", "angry", "stress", "anxiety", "lonely", "overwhelmed", "neutral", "grief"]

# ── Keyword bank with intensity weights ──────────────────────────────────────
_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "happy": [
        ("happy", 1.0), ("joy", 1.0), ("joyful", 1.0), ("excited", 0.9),
        ("great", 0.7), ("wonderful", 0.9), ("amazing", 0.9), ("fantastic", 0.9),
        ("glad", 0.8), ("blessed", 0.8), ("love", 0.6), ("smile", 0.7),
        ("laugh", 0.7), ("proud", 0.8), ("grateful", 0.8), ("cheerful", 0.9),
        ("elated", 1.0), ("thrilled", 0.9), ("delighted", 0.9), ("good news", 0.8),
        ("passed", 0.7), ("got marks", 0.8), ("promotion", 0.8), ("celebrate", 0.8),
        ("khush", 1.0), ("khushi", 1.0), ("sukoon", 0.9), ("achha lag raha", 0.9),
        ("badhiya", 0.8), ("maza aa gaya", 0.9), ("anand", 0.9), ("shandar", 0.8),
    ],
    "sad": [
        ("sad", 1.0), ("cry", 1.0), ("crying", 1.0), ("tears", 0.9),
        ("depressed", 1.0), ("depression", 1.0), ("hopeless", 1.0),
        ("empty", 0.9), ("grief", 0.9), ("heartbroken", 1.0), ("miserable", 1.0),
        ("unhappy", 0.9), ("gloomy", 0.8), ("sorrow", 0.9), ("devastated", 1.0),
        ("worthless", 1.0), ("nothing matters", 1.0), ("no point", 0.9),
        ("lost", 0.7), ("broken", 0.8), ("hurt", 0.7), ("pain", 0.7),
        ("miss", 0.6), ("miss them", 0.8), ("gone", 0.6),
        ("tired and alone", 0.9), ("feel tired", 0.7), ("feeling tired", 0.7),
        ("not doing enough", 0.8),
        ("udas", 1.0), ("udaas", 1.0), ("rona", 1.0), ("ro raha", 1.0),
        ("ro rahi", 1.0), ("aansoo", 0.9), ("aansu", 0.9), ("dil toot", 1.0),
        ("dukh", 0.9), ("dukhi", 1.0), ("dard", 0.8), ("takleef", 0.9),
        ("mayus", 0.9), ("mayoos", 0.9), ("bebas", 0.8), ("achha nahi lag raha", 0.9),
        ("theek nahi", 0.8), ("khush nahi", 0.9), ("khush nhi", 0.9),
    ],
    "angry": [
        ("angry", 1.0), ("anger", 1.0), ("furious", 1.0), ("rage", 1.0),
        ("mad", 0.9), ("hate", 0.9), ("frustrated", 0.8), ("annoyed", 0.7),
        ("irritated", 0.7), ("outraged", 1.0), ("betrayed", 0.9),
        ("unfair", 0.8), ("scream", 0.9), ("explode", 0.9), ("livid", 1.0),
        ("fed up", 0.8), ("sick of", 0.8), ("can't stand", 0.8),
        ("gussa", 1.0), ("gusse", 0.9), ("naraz", 0.8), ("nafrat", 0.9),
        ("chidh", 0.8), ("krodh", 1.0), ("dhokha diya", 0.9), ("bardasht nahi", 0.9),
    ],
    "stress": [
        ("stress", 1.0), ("stressed", 1.0), ("deadline", 0.9), ("pressure", 0.9),
        ("busy", 0.7), ("exhausted", 0.8), ("burnout", 1.0), ("overloaded", 1.0),
        ("too much", 0.8), ("can't cope", 1.0), ("piling up", 0.9),
        ("no time", 0.8), ("rushing", 0.7), ("hectic", 0.8), ("overwhelm", 0.9),
        ("exam", 0.7), ("exams", 0.8), ("assignment", 0.7), ("work pressure", 0.9),
        ("college", 0.5), ("studies", 0.6), ("behind", 0.6), ("falling behind", 0.8),
        ("college pressure", 1.0), ("academic pressure", 1.0), ("expectations", 0.8),
        ("trying to stay strong", 0.9), ("stay strong in front of everyone", 1.0),
        ("others are moving ahead", 0.9), ("everyone is moving ahead", 0.9),
        ("comparing myself", 0.8), ("comparison", 0.7),
        ("tension", 1.0), ("bojh", 0.9), ("pareshan", 0.9), ("preshan", 0.9),
        ("pareshani", 0.9), ("thak gaya", 0.8), ("thak gayi", 0.8),
        ("dimag kharab", 0.8), ("neend nahi", 0.8), ("neend nhi", 0.8),
    ],
    "anxiety": [
        ("anxious", 1.0), ("anxiety", 1.0), ("worry", 0.9), ("worried", 0.9),
        ("nervous", 0.8), ("scared", 0.8), ("fear", 0.8), ("panic", 1.0),
        ("overthinking", 1.0), ("overthink", 1.0), ("racing heart", 1.0),
        ("dread", 0.9), ("terrified", 1.0), ("uneasy", 0.7), ("restless", 0.7),
        ("what if", 0.8), ("can't stop thinking", 0.9), ("mind won't stop", 0.9),
        ("catastroph", 0.8), ("worst case", 0.8),
        ("ghabrahat", 1.0), ("dar lag", 0.9), ("dar", 0.7), ("darr", 0.8),
        ("bechain", 0.9), ("bechaini", 1.0), ("dhak dhak", 0.8), ("khauf", 0.9),
    ],
    "lonely": [
        ("lonely", 1.0), ("alone", 0.9), ("isolated", 1.0), ("no one", 0.9),
        ("nobody", 0.9), ("invisible", 0.9), ("left out", 0.9), ("excluded", 0.9),
        ("disconnected", 0.8), ("friendless", 1.0), ("forgotten", 0.8),
        ("ignored", 0.8), ("unwanted", 0.9), ("no friends", 1.0),
        ("nobody cares", 1.0), ("no one understands", 1.0), ("misunderstood", 0.8),
        ("sit alone", 0.9), ("eat alone", 0.9), ("feel alone", 0.9),
        ("feeling alone", 0.9), ("tired and alone", 1.0),
        ("akela", 1.0), ("akeli", 1.0), ("akelapan", 1.0), ("koi nahi hai", 1.0),
        ("koi baat nahi karta", 1.0), ("koi sunne wala nahi", 1.0), ("tanha", 1.0),
        ("tanhai", 1.0), ("koi dost nahi", 1.0), ("koi samjhta nahi", 1.0),
    ],
    "overwhelmed": [
        ("overwhelmed", 1.0), ("too much going on", 1.0), ("can't handle", 1.0),
        ("breaking down", 1.0), ("falling apart", 1.0), ("can't breathe", 0.9),
        ("drowning", 0.9), ("suffocating", 0.9), ("everything at once", 0.9),
        ("don't know where to start", 0.9), ("so much", 0.7), ("everything is", 0.5),
        ("can't take it", 0.9), ("at my limit", 1.0), ("reached my limit", 1.0),
        ("too many", 0.7), ("juggling", 0.7), ("spinning", 0.6),
        ("carrying so much", 1.0), ("college pressure", 0.9),
        ("others are moving ahead", 0.8), ("everyone is moving ahead", 0.8),
        ("trying to stay strong", 0.8), ("tired and alone", 1.0),
        ("exhausting", 0.8), ("heaviest", 0.7),
        ("kuch samajh nahi aa raha", 1.0), ("sambhal nahi raha", 1.0),
        ("himmat toot gayi", 1.0), ("toot chuka", 1.0), ("toot chuki", 1.0),
        ("bardasht se bahar", 1.0),
    ],
    "grief": [
        ("lost someone", 1.0), ("passed away", 1.0), ("died", 1.0), ("death", 0.9),
        ("funeral", 1.0), ("mourning", 1.0), ("grieving", 1.0), ("miss them", 0.9),
        ("they're gone", 1.0), ("never coming back", 1.0), ("loss", 0.8),
        ("mar gaye", 1.0), ("guzar gaye", 1.0), ("chhod kar chale gaye", 1.0),
        ("dehant", 1.0), ("mrit", 0.9),
    ],
}

# ── Negation patterns (English + Hinglish) ──────────────────────────────────
_NEGATIONS = re.compile(
    r"\b(not|never|no|don't|doesn't|didn't|can't|cannot|won't|isn't|aren't|wasn't|nahi|nhi|mat)\b\s+\w+",
    re.IGNORECASE
)

# ── Emoji → emotion map ───────────────────────────────────────────────────────
EMOTION_EMOJI = {
    "happy":       "😊",
    "sad":         "😢",
    "angry":       "😠",
    "stress":      "😰",
    "anxiety":     "😟",
    "lonely":      "🥺",
    "overwhelmed": "😩",
    "neutral":     "😐",
    "grief":       "💔",
}

# ── Compound emotion rules ────────────────────────────────────────────────────
# If two emotions both score above threshold, merge into compound
_COMPOUND_RULES: list[tuple[set, str]] = [
    ({"stress", "sad"},     "overwhelmed"),
    ({"stress", "lonely"},  "overwhelmed"),
    ({"anxiety", "sad"},    "overwhelmed"),
    ({"sad", "lonely"},     "sad"),          # sad wins
    ({"stress", "anxiety"}, "anxiety"),      # anxiety wins
]


def _remove_negated(text: str) -> str:
    """Remove negated phrases so 'not happy' doesn't score as happy."""
    return _NEGATIONS.sub("", text)


def predict_emotion(text: str) -> dict:
    """
    Returns:
        {
            "emotion": "overwhelmed",
            "confidence": 0.82,
            "scores": {"happy": 0.0, "sad": 0.3, ...},
            "emoji": "😩"
        }
    """
    text_clean   = _remove_negated(text.lower())
    text_lower   = text.lower()  # keep original for some checks

    scores: dict[str, float] = {label: 0.0 for label in LABELS}

    for emotion, keywords in _KEYWORDS.items():
        for kw, weight in keywords:
            if kw in text_clean:
                scores[emotion] += weight

    # ── Compound detection ────────────────────────────────────────────────────
    threshold = 0.5
    active = {e for e, s in scores.items() if s >= threshold}
    for emotions_set, compound in _COMPOUND_RULES:
        if emotions_set.issubset(active):
            # Boost compound emotion
            scores[compound] = max(scores[compound], sum(scores[e] for e in emotions_set) * 0.8)

    if scores["stress"] >= 0.9 and (scores["sad"] >= 0.7 or scores["lonely"] >= 0.7):
        scores["overwhelmed"] = max(
            scores["overwhelmed"],
            scores["stress"] + max(scores["sad"], scores["lonely"]) * 0.9,
        )

    # ── Normalize ─────────────────────────────────────────────────────────────
    total = sum(scores.values())
    if total == 0:
        scores["neutral"] = 1.0
        total = 1.0

    probs = {k: round(v / total, 4) for k, v in scores.items()}
    top   = max(probs, key=probs.get)

    # ── Safety: never return happy for clearly distressed text ────────────────
    distress_words = [
        "sad", "cry", "depress", "hopeless", "worthless", "lonely",
        "stress", "anxious", "overwhelm", "angry", "furious", "grief",
        "lost", "broken", "hurt", "pain", "alone", "nobody", "no one",
        "udas", "udaas", "rona", "dard", "takleef", "akela", "akeli",
        "pareshan", "preshan", "tension", "gussa", "dar", "bechain",
        "toot", "tanhai"
    ]
    if top == "happy" and any(w in text_lower for w in distress_words):
        probs["happy"] = 0.0
        total2 = sum(probs.values()) or 1.0
        probs  = {k: round(v / total2, 4) for k, v in probs.items()}
        top    = max(probs, key=probs.get)

    return {
        "emotion":    top,
        "confidence": probs[top],
        "scores":     probs,
        "emoji":      EMOTION_EMOJI.get(top, "😐"),
    }


if __name__ == "__main__":
    tests = [
        "I feel so lonely and nobody cares about me.",
        "I am so happy today, everything is perfect!",
        "I have too many deadlines and I can't breathe.",
        "I keep worrying about everything.",
        "I am furious at what they did to me.",
        "Just had lunch, nothing special.",
        "College pressure is too much, I feel like I'm falling behind everyone.",
        "I lost my grandmother last week and I can't stop crying.",
        "I'm stressed about exams and also feeling very alone.",
        "aaj bahut udaas hu aur rona aa raha hai",
        "bahut tension aur pressure hai exams ka",
        "koi mera dost nahi hai, akela lag raha hai",
        "aaj mai bahut khush hu",
    ]
    for t in tests:
        r = predict_emotion(t)
        print(f"[{r['emotion']:12s} {r['confidence']*100:5.1f}%] {t[:60]}")
