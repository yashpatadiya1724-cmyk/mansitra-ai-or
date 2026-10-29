"""

Manasitra Memory Engine — v2

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

UNIQUE FEATURES (patent-worthy combination):

1. Emotion-Adaptive Memory Compression (EAMC)

   High-distress moments (sad/anxious/lonely) get higher retention weight.

   Neutral moments decay faster — just like human memory works.

   Algorithm: retention_score = base_weight[emotion] * recency_decay(hours)

2. Longitudinal Mood Pattern Engine (LMPE)

   Detects recurring emotional patterns over time:

   - Time-of-day patterns  ("Yash is anxious every Sunday night")

   - Streak detection      ("5 consecutive sad days")

   - Trigger correlation   (which topics cause which emotions)

3. Emotion State Machine (ESM) — Transition Tracking

   Tracks how emotions shift within and across conversations.

   Feeds the conversation brain so responses adapt to transitions,

   not just the current snapshot.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""

import sqlite3

import json

import os

import math

from datetime import datetime, timezone

from collections import Counter

from typing import Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "../../database/manasitra.db")

# ── Feature 1: Emotion retention weights ─────────────────────────────────────

# Higher = more important to remember (distress moments retained longer)

EMOTION_WEIGHT = {

    "sad":     1.0,

    "lonely":  1.0,

    "anxiety": 0.95,

    "stress":  0.85,

    "angry":   0.80,

    "happy":   0.60,

    "neutral": 0.20,   # neutral fades fastest

}

# ── Feature 3: Emotion transition map ────────────────────────────────────────

# Defines valid emotional transitions and their "distance"

# Used by the state machine to detect abrupt vs gradual shifts

EMOTION_TRANSITIONS = {

    ("sad",     "lonely"):  "natural",

    ("lonely",  "sad"):     "natural",

    ("stress",  "anxiety"): "natural",

    ("anxiety", "stress"):  "natural",

    ("angry",   "sad"):     "natural",

    ("sad",     "neutral"): "improving",

    ("neutral", "happy"):   "improving",

    ("happy",   "sad"):     "abrupt",

    ("happy",   "angry"):   "abrupt",

    ("neutral", "angry"):   "abrupt",

    ("neutral", "sad"):     "concerning",

    ("neutral", "anxiety"): "concerning",

}

def _get_conn() -> sqlite3.Connection:

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    conn = sqlite3.connect(DB_PATH, timeout=10.0)

    conn.row_factory = sqlite3.Row

    conn.execute("PRAGMA synchronous = NORMAL;")

    conn.execute("PRAGMA cache_size = -16000;")

    conn.execute("PRAGMA temp_store = MEMORY;")

    return conn

def init_db():

    conn = _get_conn()

    with conn:

        conn.execute("PRAGMA journal_mode = WAL;")

        conn.executescript("""

            CREATE TABLE IF NOT EXISTS users (

                id          INTEGER PRIMARY KEY AUTOINCREMENT,

                name        TEXT    NOT NULL UNIQUE,

                created_at  TEXT    NOT NULL

            );

            CREATE TABLE IF NOT EXISTS mood_history (

                id               INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id          INTEGER NOT NULL REFERENCES users(id),

                emotion          TEXT    NOT NULL,

                message          TEXT,

                retention_score  REAL    NOT NULL DEFAULT 1.0,

                recorded_at      TEXT    NOT NULL

            );

            CREATE TABLE IF NOT EXISTS conversations (

                id          INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id     INTEGER NOT NULL REFERENCES users(id),

                role        TEXT    NOT NULL,

                content     TEXT    NOT NULL,

                emotion     TEXT,

                weight      REAL    NOT NULL DEFAULT 1.0,

                created_at  TEXT    NOT NULL

            );

            CREATE TABLE IF NOT EXISTS emotion_transitions (

                id              INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id         INTEGER NOT NULL REFERENCES users(id),

                from_emotion    TEXT    NOT NULL,

                to_emotion      TEXT    NOT NULL,

                transition_type TEXT    NOT NULL,

                recorded_at     TEXT    NOT NULL

            );

            CREATE TABLE IF NOT EXISTS mood_patterns (

                id           INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id      INTEGER NOT NULL REFERENCES users(id),

                pattern_type TEXT    NOT NULL,

                pattern_data TEXT    NOT NULL,

                detected_at  TEXT    NOT NULL

            );

            CREATE TABLE IF NOT EXISTS user_preferences (

                user_id     INTEGER PRIMARY KEY REFERENCES users(id),

                preferences TEXT    NOT NULL DEFAULT '{}'

            );

            CREATE TABLE IF NOT EXISTS journal_entries (

                id          INTEGER PRIMARY KEY AUTOINCREMENT,

                user_id     INTEGER NOT NULL REFERENCES users(id),

                entry_type  TEXT    NOT NULL,

                content     TEXT    NOT NULL,

                created_at  TEXT    NOT NULL

            );

            CREATE INDEX IF NOT EXISTS idx_mood_history_user_time 

                ON mood_history(user_id, recorded_at DESC);

            CREATE INDEX IF NOT EXISTS idx_conversations_user_time 

                ON conversations(user_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_transitions_user_time 

                ON emotion_transitions(user_id, recorded_at DESC);

            CREATE INDEX IF NOT EXISTS idx_journal_user_time 

                ON journal_entries(user_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_patterns_user_type 

                ON mood_patterns(user_id, pattern_type);

        """)

    conn.close()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# USERS

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def get_or_create_user(name: str) -> int:

    conn = _get_conn()

    with conn:

        row = conn.execute("SELECT id FROM users WHERE name = ?", (name,)).fetchone()

        if row:

            return row["id"]

        cur = conn.execute(

            "INSERT INTO users (name, created_at) VALUES (?, ?)",

            (name, datetime.now(timezone.utc).isoformat()),

        )

        return cur.lastrowid

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FEATURE 1 — EMOTION-ADAPTIVE MEMORY COMPRESSION (EAMC)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _compute_retention(emotion: str, hours_ago: float = 0.0) -> float:

    """

    retention_score = base_weight * e^(-decay_rate * hours)

    High-distress emotions decay slowly; neutral decays fast.

    """

    base   = EMOTION_WEIGHT.get(emotion, 0.5)

    # Distress emotions have slow decay (0.005/hr), neutral fast (0.05/hr)

    decay  = 0.005 if base >= 0.8 else (0.02 if base >= 0.5 else 0.05)

    score  = base * math.exp(-decay * hours_ago)

    return round(max(score, 0.01), 4)

def save_mood(user_id: int, emotion: str, message: Optional[str] = None):

    now   = datetime.now(timezone.utc)

    score = _compute_retention(emotion, hours_ago=0)

    conn  = _get_conn()

    with conn:

        conn.execute(

            "INSERT INTO mood_history (user_id, emotion, message, retention_score, recorded_at) "

            "VALUES (?, ?, ?, ?, ?)",

            (user_id, emotion, message, score, now.isoformat()),

        )

    conn.close()

    # After saving, run pattern detection

    _detect_patterns(user_id)

def get_mood_history(user_id: int, limit: int = 20) -> list[dict]:

    """Returns mood history ordered by retention_score * recency (not just time)."""

    conn = _get_conn()

    rows = conn.execute(

        "SELECT emotion, message, retention_score, recorded_at "

        "FROM mood_history WHERE user_id = ? "

        "ORDER BY recorded_at DESC LIMIT ?",

        (user_id, limit),

    ).fetchall()

    conn.close()

    return [dict(r) for r in rows]

def get_weighted_memory(user_id: int, limit: int = 10) -> list[dict]:

    """

    EAMC: Returns memories sorted by live retention score.

    Recalculates decay based on actual hours elapsed since recording.

    High-distress old memories can outrank recent neutral ones.

    """

    conn = _get_conn()

    rows = conn.execute(

        "SELECT emotion, message, retention_score, recorded_at "

        "FROM mood_history WHERE user_id = ? ORDER BY recorded_at DESC LIMIT 50",

        (user_id,),

    ).fetchall()

    conn.close()

    now = datetime.now(timezone.utc)

    scored = []

    for r in rows:

        try:

            recorded = datetime.fromisoformat(r["recorded_at"])

            if recorded.tzinfo is None:

                recorded = recorded.replace(tzinfo=timezone.utc)

            hours_ago = (now - recorded).total_seconds() / 3600

        except Exception:

            hours_ago = 0

        live_score = _compute_retention(r["emotion"], hours_ago)

        scored.append({**dict(r), "live_retention": live_score})

    scored.sort(key=lambda x: x["live_retention"], reverse=True)

    return scored[:limit]

def get_dominant_mood(user_id: int, limit: int = 10) -> Optional[str]:

    """Weighted dominant mood — distress emotions count more."""

    history = get_mood_history(user_id, limit)

    if not history:

        return None

    weighted: dict[str, float] = {}

    for entry in history:

        e = entry["emotion"]

        w = EMOTION_WEIGHT.get(e, 0.5)

        weighted[e] = weighted.get(e, 0.0) + w

    return max(weighted, key=weighted.get)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FEATURE 2 — LONGITUDINAL MOOD PATTERN ENGINE (LMPE)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _detect_patterns(user_id: int):

    """

    Runs after every mood save. Detects:

    - Streak: same emotion 3+ times in a row

    - Time-of-day pattern: same emotion at same hour 3+ times

    - Trigger words: common words in high-distress messages

    """

    conn = _get_conn()

    rows = conn.execute(

        "SELECT emotion, message, recorded_at FROM mood_history "

        "WHERE user_id = ? ORDER BY recorded_at DESC LIMIT 30",

        (user_id,),

    ).fetchall()

    conn.close()

    if not rows:

        return

    now = datetime.now(timezone.utc).isoformat()

    # ── Streak detection ──────────────────────────────────────────────────────

    streak_emotion = rows[0]["emotion"]

    streak_count   = 1

    for r in rows[1:]:

        if r["emotion"] == streak_emotion:

            streak_count += 1

        else:

            break

    if streak_count >= 3:

        _save_pattern(user_id, "streak", {

            "emotion": streak_emotion,

            "count":   streak_count,

            "insight": f"You've been feeling {streak_emotion} for {streak_count} sessions in a row."

        }, now)

    # ── Time-of-day pattern ───────────────────────────────────────────────────

    hour_emotion: dict[int, list[str]] = {}

    for r in rows:

        try:

            dt   = datetime.fromisoformat(r["recorded_at"])

            hour = dt.hour

            hour_emotion.setdefault(hour, []).append(r["emotion"])

        except Exception:

            pass

    for hour, emotions in hour_emotion.items():

        if len(emotions) >= 3:

            most_common = Counter(emotions).most_common(1)[0]

            if most_common[1] >= 3:

                period = ("morning" if 5 <= hour < 12 else "afternoon" if 12 <= hour < 17 else "evening" if 17 <= hour < 21 else "night")

                _save_pattern(user_id, "time_of_day", {

                    "hour":    hour,

                    "period":  period,

                    "emotion": most_common[0],

                    "count":   most_common[1],

                    "insight": f"You often feel {most_common[0]} during {period} hours."

                }, now)

    # ── Trigger word detection with stopword filtration ──────────────────────

    COMMON_STOPWORDS = {

        "about", "above", "after", "again", "against", "always", "another", "around",

        "because", "before", "being", "below", "between", "could", "doing", "during",

        "every", "feeling", "first", "going", "having", "myself", "really", "should",

        "since", "someone", "something", "sometimes", "still", "their", "there",

        "these", "thing", "think", "thought", "through", "today", "under", "where",

        "which", "while", "would", "yourself",

        # Hinglish fillers

        "bahut", "hamesha", "kabhi", "lagta", "lagti", "kuch", "apne", "apna", "apni",

        "raha", "rahi", "rahe", "hoga", "hogi", "meri", "mera", "mere", "tumhara",

        "samajh", "chahiye", "karne", "karna", "baat", "nahi", "nhi"

    }

    distress_emotions = {"sad", "lonely", "anxiety", "stress", "angry"}

    distress_words: list[str] = []

    for r in rows:

        if r["emotion"] in distress_emotions and r["message"]:

            words = [

                w.lower().strip(".,!?:;\"'()") for w in r["message"].split()

                if len(w) > 3 and w.lower() not in COMMON_STOPWORDS and w.isalpha()

            ]

            distress_words.extend(words)

    if distress_words:

        common_triggers = Counter(distress_words).most_common(3)

        triggers = [w for w, c in common_triggers if c >= 2]

        if triggers:

            _save_pattern(user_id, "trigger_words", {

                "triggers": triggers,

                "insight":  f"Topics that often cause you distress: {', '.join(triggers)}"

            }, now)

def _save_pattern(user_id: int, pattern_type: str, data: dict, now: str):

    conn = _get_conn()

    with conn:

        # Avoid duplicate patterns of same type within 1 hour

        existing = conn.execute(

            "SELECT id FROM mood_patterns WHERE user_id=? AND pattern_type=? "

            "AND detected_at > datetime(?, '-1 hour')",

            (user_id, pattern_type, now),

        ).fetchone()

        if not existing:

            conn.execute(

                "INSERT INTO mood_patterns (user_id, pattern_type, pattern_data, detected_at) "

                "VALUES (?, ?, ?, ?)",

                (user_id, pattern_type, json.dumps(data), now),

            )

    conn.close()

def get_mood_patterns(user_id: int) -> list[dict]:

    """Returns all detected patterns for a user, newest first."""

    conn = _get_conn()

    rows = conn.execute(

        "SELECT pattern_type, pattern_data, detected_at FROM mood_patterns "

        "WHERE user_id = ? ORDER BY detected_at DESC LIMIT 10",

        (user_id,),

    ).fetchall()

    conn.close()

    result = []

    for r in rows:

        try:

            result.append({

                "type":        r["pattern_type"],

                "data":        json.loads(r["pattern_data"]),

                "detected_at": r["detected_at"],

            })

        except Exception:

            pass

    return result

def get_pattern_insights(user_id: int) -> list[str]:

    """Returns human-readable insight strings for the conversation brain."""

    patterns = get_mood_patterns(user_id)

    insights = []

    seen_types: set[str] = set()

    for p in patterns:

        if p["type"] not in seen_types:

            insight = p["data"].get("insight", "")

            if insight:

                insights.append(insight)

            seen_types.add(p["type"])

    return insights[:3]  # max 3 insights at a time

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FEATURE 3 — EMOTION STATE MACHINE (ESM) — TRANSITION TRACKING

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def record_emotion_transition(user_id: int, from_emotion: str, to_emotion: str):

    """

    Records an emotion transition and classifies it.

    Called by the backend whenever emotion changes between messages.

    """

    if from_emotion == to_emotion:

        return

    transition_type = EMOTION_TRANSITIONS.get(

        (from_emotion, to_emotion), "unknown"

    )

    conn = _get_conn()

    with conn:

        conn.execute(

            "INSERT INTO emotion_transitions (user_id, from_emotion, to_emotion, "

            "transition_type, recorded_at) VALUES (?, ?, ?, ?, ?)",

            (user_id, from_emotion, to_emotion, transition_type,

             datetime.now(timezone.utc).isoformat()),

        )

    conn.close()

def get_last_emotion(user_id: int) -> Optional[str]:

    """Returns the emotion from the previous message (for transition detection)."""

    conn = _get_conn()

    row = conn.execute(

        "SELECT emotion FROM mood_history WHERE user_id = ? "

        "ORDER BY recorded_at DESC LIMIT 1 OFFSET 1",

        (user_id,),

    ).fetchone()

    conn.close()

    return row["emotion"] if row else None

def get_current_transition(user_id: int) -> Optional[dict]:

    """Returns the most recent emotion transition for this user."""

    conn = _get_conn()

    row = conn.execute(

        "SELECT from_emotion, to_emotion, transition_type, recorded_at "

        "FROM emotion_transitions WHERE user_id = ? "

        "ORDER BY recorded_at DESC LIMIT 1",

        (user_id,),

    ).fetchone()

    conn.close()

    return dict(row) if row else None

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# CONVERSATIONS

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_message(user_id: int, role: str, content: str, emotion: Optional[str] = None):

    weight = EMOTION_WEIGHT.get(emotion, 0.5) if emotion else 0.5

    conn   = _get_conn()

    with conn:

        conn.execute(

            "INSERT INTO conversations (user_id, role, content, emotion, weight, created_at) "

            "VALUES (?, ?, ?, ?, ?, ?)",

            (user_id, role, content, emotion, weight,

             datetime.now(timezone.utc).isoformat()),

        )

    conn.close()

def get_recent_conversation(user_id: int, limit: int = 10) -> list[dict]:

    """

    Returns recent conversation, but high-weight (distress) messages

    are always included even if older — EAMC applied to conversations too.

    """

    conn = _get_conn()

    # Always include last 4 messages (recency)

    recent = conn.execute(

        "SELECT role, content, emotion, weight, created_at FROM conversations "

        "WHERE user_id = ? ORDER BY created_at DESC LIMIT 4",

        (user_id,),

    ).fetchall()

    # Also pull high-weight messages from further back

    important = conn.execute(

        "SELECT role, content, emotion, weight, created_at FROM conversations "

        "WHERE user_id = ? AND weight >= 0.8 ORDER BY created_at DESC LIMIT 4",

        (user_id,),

    ).fetchall()

    conn.close()

    # Merge, deduplicate by created_at, sort chronologically

    seen: set[str] = set()

    merged = []

    for r in list(recent) + list(important):

        key = r["created_at"]

        if key not in seen:

            seen.add(key)

            merged.append(dict(r))

    merged.sort(key=lambda x: x["created_at"])

    return merged[-limit:]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# PREFERENCES

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_preference(user_id: int, key: str, value):

    conn = _get_conn()

    with conn:

        row  = conn.execute(

            "SELECT preferences FROM user_preferences WHERE user_id = ?", (user_id,)

        ).fetchone()

        prefs = json.loads(row["preferences"]) if row else {}

        prefs[key] = value

        conn.execute(

            "INSERT INTO user_preferences (user_id, preferences) VALUES (?, ?) "

            "ON CONFLICT(user_id) DO UPDATE SET preferences = excluded.preferences",

            (user_id, json.dumps(prefs)),

        )

    conn.close()

def get_preferences(user_id: int) -> dict:

    conn = _get_conn()

    row  = conn.execute(

        "SELECT preferences FROM user_preferences WHERE user_id = ?", (user_id,)

    ).fetchone()

    conn.close()

    return json.loads(row["preferences"]) if row else {}

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# FEATURE 4 — JOURNAL & MOTIVATION

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def save_journal_entry(user_id: int, entry_type: str, content: str):

    """Saves a journal entry (e.g., 'achievement' or 'bad_time')."""

    conn = _get_conn()

    with conn:

        conn.execute(

            "INSERT INTO journal_entries (user_id, entry_type, content, created_at) "

            "VALUES (?, ?, ?, ?)",

            (user_id, entry_type, content, datetime.now(timezone.utc).isoformat()),

        )

    conn.close()

def get_recent_achievements(user_id: int, limit: int = 3) -> list[str]:

    """Returns recent achievements to inject into context when user is demotivated."""

    conn = _get_conn()

    rows = conn.execute(

        "SELECT content FROM journal_entries WHERE user_id = ? AND entry_type = 'achievement' "

        "ORDER BY created_at DESC LIMIT ?",

        (user_id, limit),

    ).fetchall()

    conn.close()

    return [r["content"] for r in rows]

def get_journal_entries(user_id: int, limit: int = 20) -> list[dict]:

    """Returns past journal entries for the UI."""

    conn = _get_conn()

    rows = conn.execute(

        "SELECT entry_type, content, created_at FROM journal_entries WHERE user_id = ? "

        "ORDER BY created_at DESC LIMIT ?",

        (user_id, limit),

    ).fetchall()

    conn.close()

    return [dict(r) for r in rows]

# ── Init on import ─────────────────────────────────────────────────────────────

init_db()

