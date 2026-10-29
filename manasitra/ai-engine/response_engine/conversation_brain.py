"""

Manasitra Conversation Brain — v3

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Priority:

  1. Custom trained Manasitra model (no LLM, no Ollama)

  2. Ollama fallback (if custom model not trained yet)

  3. Rule-based fallback (if both unavailable)



STRICT RULES enforced in every prompt:

  1. Detect emotion from context

  2. Validate feelings first — never skip this

  3. Never assume facts not mentioned by user

  4. Never force a diagnosis

  5. Always ask ONE follow-up question

  6. Match emoji to emotion — never use 😊 for sad/stressed input

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

"""



import os

import sys

import random

import requests

import json

from typing import Optional, Generator



# Add custom_model to path

_CUSTOM_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "custom_model"))

if _CUSTOM_MODEL_DIR not in sys.path:

    sys.path.insert(0, _CUSTOM_MODEL_DIR)



# ── Ollama config ─────────────────────────────────────────────────────────────

OLLAMA_BASE  = os.getenv("OLLAMA_URL", "http://localhost:11434")

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "manasitra")

CHAT_URL     = f"{OLLAMA_BASE}/api/chat"



# ── Custom model ──────────────────────────────────────────────────────────────

_CUSTOM_MODEL_STATUS = {"available": None, "epoch": 0, "mtime": 0.0}

def get_custom_model_info() -> dict:
    """Returns cached custom model metadata without redundant disk loads: {"ready": bool, "epoch": int}."""
    ckpt_path = os.path.join(os.path.dirname(__file__), "../custom_model/saved_model/manasitra_model.pt")
    if not os.path.exists(ckpt_path):
        return {"ready": False, "epoch": 0}
    try:
        from inference import is_model_trained  # type: ignore
        if not is_model_trained():
            return {"ready": False, "epoch": 0}
        mtime = os.path.getmtime(ckpt_path)
        if _CUSTOM_MODEL_STATUS["available"] is not None and _CUSTOM_MODEL_STATUS["mtime"] == mtime:
            return {"ready": _CUSTOM_MODEL_STATUS["available"], "epoch": _CUSTOM_MODEL_STATUS.get("epoch", 0)}

        import torch
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        epoch = ckpt.get("epoch", 0)
        min_epoch = int(os.getenv("CUSTOM_MODEL_MIN_EPOCH", "5"))
        is_avail = epoch >= min_epoch
        _CUSTOM_MODEL_STATUS["available"] = is_avail
        _CUSTOM_MODEL_STATUS["epoch"] = epoch
        _CUSTOM_MODEL_STATUS["mtime"] = mtime
        return {"ready": is_avail, "epoch": epoch}
    except Exception:
        return {"ready": False, "epoch": 0}

def _custom_model_available() -> bool:
    return get_custom_model_info()["ready"]

def _call_custom_model(text: str, emotion: str) -> Optional[str]:

    """Call our own trained Seq2Seq model — no LLM, no Ollama."""

    try:

        from inference import generate  # type: ignore

        reply = generate(text, emotion)

        return reply if reply and len(reply.strip()) > 10 else None

    except Exception:

        return None



# ── Emotion hints ─────────────────────────────────────────────────────────────

EMOTION_HINT = {

    "happy":       "feeling happy and positive",

    "sad":         "feeling sad and needs comfort",

    "angry":       "feeling angry and needs calm validation",

    "stress":      "feeling stressed and overwhelmed",

    "anxiety":     "feeling anxious and needs grounding",

    "lonely":      "feeling lonely and needs connection",

    "overwhelmed": "feeling overwhelmed — too much at once",

    "grief":       "grieving a loss and needs gentle presence",

    "neutral":     "in a neutral mood",

}



# ── Emotion-correct emoji ─────────────────────────────────────────────────────

EMOTION_EMOJI = {

    "happy":       "😊",

    "sad":         "😢",

    "angry":       "😠",

    "stress":      "😰",

    "anxiety":     "😟",

    "lonely":      "🥺",

    "overwhelmed": "😩",

    "grief":       "💔",

    "neutral":     "😐",

}



# ── Feature 3: Transition-aware response modifiers ───────────────────────────

TRANSITION_GUIDANCE = {

    "abrupt":     "NOTE: The user's emotion just shifted abruptly. Gently acknowledge this shift before responding.",

    "concerning": "NOTE: The user moved from neutral to distressed. Show extra care and ask what triggered this.",

    "improving":  "NOTE: The user seems slightly better. Gently acknowledge this positive shift.",

    "natural":    "",

}



# ── Strict response rules injected into every prompt ─────────────────────────

STRICT_RULES = """

STRICT RULES — follow every single one:

1. VALIDATE first — acknowledge the emotion before anything else

2. NEVER assume facts the user didn't mention (no wife, no family, no job unless they said so)

3. NEVER diagnose — do not say "serious mental illness", "depression", "disorder"

4. NEVER suggest another professional unless user asks or it's a crisis

5. Ask exactly ONE analytical follow-up question at the end to explore cognitive/behavioral patterns

6. Keep response to 3-5 sentences maximum

7. Use the correct emoji for the emotion — NEVER use 😊 for sad/stressed/angry/lonely input

8. Do NOT end with a happy emoji if the user is distressed

9. Sound like a professional, analytical, and empathetic clinical psychologist

10. Never say "As an AI" or "I am a language model"

"""



# ── Fallback responses ────────────────────────────────────────────────────────

FALLBACK = {

    "happy":       ["That's wonderful! What made today so special for you? 😊"],

    "sad":         ["I hear you, and I'm really sorry you're feeling this way. I'm here with you. 💙 What's been weighing on you?"],

    "angry":       ["That would make anyone angry. Your feelings are completely valid. 😠 Want to tell me what happened?"],

    "stress":      ["That's a lot to carry right now. 😰 What feels most urgent to you at this moment?"],

    "anxiety":     ["You're safe right now. Take one slow breath. 😟 What's the one thing worrying you most?"],

    "lonely":      ["I'm here, and I'm glad you reached out. You are not alone. 🥺 What's been going on lately?"],

    "overwhelmed": ["I can hear how much you're carrying right now. 😩 What part feels heaviest — is it the pressure, the expectations, or something else?"],

    "grief":       ["I'm so sorry for your loss. Grief is love with nowhere to go. 💔 Would you like to tell me about them?"],

    "neutral":     ["I'm here whenever you need me. How are you really doing today?"],

}



# ── Crisis detection ──────────────────────────────────────────────────────────

CRISIS_KEYWORDS = [

    "kill myself", "end my life", "want to die", "suicide",

    "hurt myself", "self harm", "no reason to live", "give up on life",

    "don't want to be here", "better off dead",

    "khatam kar loon", "khatam kar lena", "jeena nahi chahta", "jeena nahi chahti",

    "mar jaana chahta", "mar jaana chahti", "zindagi khatam", "jaan de dunga",

    "jaan de dungi", "khudkhushi", "khudkushi", "zeher", "marne ka man",

    "atmahatya", "atma hatya", "koi fayda nahi jeene ka",

]



CRISIS_RESPONSE = (

    "I'm really worried about you right now, and I care about you deeply. 💙\n\n"

    "Please reach out for help immediately:\n"

    "🆘 iCall (India): 9152987821\n"

    "🆘 Vandrevala Foundation: 1860-2662-345\n"

    "🆘 AASRA: 9820466627\n\n"

    "You are not alone. Your life matters. Please call one of these numbers right now."

)





def is_crisis(text: str) -> bool:

    t = text.lower()

    return any(kw in t for kw in CRISIS_KEYWORDS)





DISTRESSED_EMOTIONS = {"sad", "stress", "anxiety", "lonely", "overwhelmed", "grief", "angry"}

HAPPY_EMOJI_MARKERS = {"😊", "😄", "😁", "ðŸ˜Š", "ðŸ˜„", "ðŸ˜"}

UNASKED_HELP_WORDS = {"counselor", "counsellor", "therapist", "psychiatrist"}

DIAGNOSIS_PHRASES = {

    "serious mental illness",

    "mental illness",

    "clinical depression",

    "you have depression",

    "you are depressed",

    "disorder",

    "diagnosis",

}

ASSUMPTION_PHRASES = {

    "your wife",

    "your husband",

    "your girlfriend",

    "your boyfriend",

    "your parents",

    "your family",

    "your job",

}





def _user_asked_for_professional_help(text: str) -> bool:

    t = text.lower()

    return any(word in t for word in UNASKED_HELP_WORDS) or "professional help" in t





def _reply_breaks_emotional_rules(reply: str, emotion: str, user_message: str) -> bool:

    """Reject replies that hallucinate facts, diagnose, or mismatch distress tone."""

    r = reply.lower()

    if any(phrase in r for phrase in ASSUMPTION_PHRASES):

        return True

    if any(phrase in r for phrase in DIAGNOSIS_PHRASES):

        return True

    if (

        not is_crisis(user_message)

        and not _user_asked_for_professional_help(user_message)

        and any(word in r for word in UNASKED_HELP_WORDS)

    ):

        return True

    if emotion in DISTRESSED_EMOTIONS and any(marker in reply for marker in HAPPY_EMOJI_MARKERS):

        return True

    return False





def _safe_fallback(emotion: str) -> str:

    return random.choice(FALLBACK.get(emotion, FALLBACK["neutral"]))





# ── Ollama helpers ────────────────────────────────────────────────────────────

_OLLAMA_CACHE = {"status": None, "model": None, "last_check": 0.0}
_OLLAMA_TTL   = 15.0  # seconds

def _check_ollama() -> tuple[bool, Optional[str]]:
    import time
    now = time.time()
    if now - _OLLAMA_CACHE["last_check"] < _OLLAMA_TTL and _OLLAMA_CACHE["status"] is not None:
        return _OLLAMA_CACHE["status"], _OLLAMA_CACHE["model"]

    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=0.8)
        if r.status_code == 200:
            models = [m["name"] for m in r.json().get("models", [])]
            chosen = None
            if models:
                for m in models:
                    if OLLAMA_MODEL in m:
                        chosen = m
                        break
                if not chosen:
                    chosen = models[0]
            _OLLAMA_CACHE["status"] = True
            _OLLAMA_CACHE["model"] = chosen
            _OLLAMA_CACHE["last_check"] = now
            return True, chosen
    except Exception:
        pass

    _OLLAMA_CACHE["status"] = False
    _OLLAMA_CACHE["model"] = None
    _OLLAMA_CACHE["last_check"] = now
    return False, None


def ollama_is_running() -> bool:
    return _check_ollama()[0]


def get_available_model() -> Optional[str]:
    return _check_ollama()[1]

def _build_messages(

    user_message:   str,

    emotion:        str,

    history:        list[dict],

    user_name:      str,

    dominant_mood:  Optional[str],

    pattern_insights: list[str],

    weighted_memory:  list[dict],

    transition:       Optional[dict],

    achievements:     list[str] = [],

    past_journals:    list[dict] = [],

) -> list[dict]:

    messages = []



    hint  = EMOTION_HINT.get(emotion, "in an unknown emotional state")

    emoji = EMOTION_EMOJI.get(emotion, "😐")



    context_lines = [

        "You are Manasitra, a warm empathetic AI psychologist.",

        "",

        f"User: {user_name} | Detected emotion: {emotion} ({hint}) {emoji}",

    ]



    if dominant_mood and dominant_mood != emotion:

        context_lines.append(f"Recent dominant mood: {dominant_mood} — keep this in mind.")



    # Feature 1 — EAMC

    if weighted_memory:

        top = [m for m in weighted_memory[:2] if m.get("message") and m["live_retention"] >= 0.4]

        if top:

            mem = " | ".join(f'[{m["emotion"]}] "{m["message"][:50]}"' for m in top)

            context_lines.append(f"Important past context: {mem}")



    # Feature 2 — LMPE

    if pattern_insights:

        context_lines.append("Patterns detected: " + " | ".join(pattern_insights))



    # Feature 3 — ESM

    if transition:

        guidance = TRANSITION_GUIDANCE.get(transition.get("transition_type", ""), "")

        if guidance:

            context_lines.append(guidance)



    # Feature 4 - Journal Motivation

    if achievements:

        context_lines.append("User's past achievements: " + " | ".join(achievements))

        context_lines.append("NOTE: User is feeling demotivated/distressed. Mention their achievements gently to motivate them.")



    if past_journals:

        journal_texts = [f"[{j.get('created_at', '')[:10]}] {j.get('content', '')}" for j in past_journals]

        context_lines.append("User's recent journal entries for context: " + " | ".join(journal_texts))

        context_lines.append("Use this deep context gently to understand their pain points, but do not aggressively quote it back to them.")



    context_lines.append("")

    context_lines.append(STRICT_RULES)



    messages.append({"role": "system", "content": "\n".join(context_lines)})



    for msg in history[-6:]:

        messages.append({"role": msg["role"], "content": msg["content"]})



    if not history or history[-1]["content"] != user_message:

        messages.append({"role": "user", "content": user_message})



    return messages





# ── Ollama calls ──────────────────────────────────────────────────────────────

def _call_ollama(messages: list[dict], model: str) -> Optional[str]:

    try:

        resp = requests.post(

            CHAT_URL,

            json={"model": model, "messages": messages, "stream": False},

            timeout=120,

        )

        resp.raise_for_status()

        return resp.json()["message"]["content"].strip()

    except Exception:

        return None





def _call_ollama_stream(messages: list[dict], model: str) -> Generator[str, None, None]:

    with requests.post(

        CHAT_URL,

        json={"model": model, "messages": messages, "stream": True},

        stream=True,

        timeout=120,

    ) as resp:

        resp.raise_for_status()

        for line in resp.iter_lines():

            if line:

                chunk = json.loads(line)

                token = chunk.get("message", {}).get("content", "")

                if token:

                    yield token

                if chunk.get("done"):

                    break





# ── Public API ────────────────────────────────────────────────────────────────

def generate_response(

    user_message:     str,

    emotion:          str,

    history:          list[dict],

    user_name:        str = "Friend",

    dominant_mood:    Optional[str] = None,

    pattern_insights: list[str] = [],

    weighted_memory:  list[dict] = [],

    transition:       Optional[dict] = None,

    achievements:     list[str] = [],

    past_journals:    list[dict] = [],

) -> str:

    if is_crisis(user_message):

        return CRISIS_RESPONSE



    # 1. Try our own custom trained model first

    if _custom_model_available():

        # Inject achievements and journals into custom model input

        if achievements or past_journals:

            context_parts = []

            if achievements:

                context_parts.append("ACHIEVEMENTS: " + " | ".join(achievements))

            if past_journals:

                j_texts = [j.get("content", "") for j in past_journals]

                context_parts.append("JOURNALS: " + " | ".join(j_texts))

            

            user_message = f"{user_message} [{' --- '.join(context_parts)}]"

            

        reply = _call_custom_model(user_message, emotion)

        if reply:

            if not _reply_breaks_emotional_rules(reply, emotion, user_message):

                return reply

            return _safe_fallback(emotion)



    # 2. Fallback to Ollama

    model = get_available_model()

    if model:

        messages = _build_messages(

            user_message, emotion, history, user_name,

            dominant_mood, pattern_insights, weighted_memory, transition, achievements, past_journals

        )

        reply = _call_ollama(messages, model)

        if reply:

            if not _reply_breaks_emotional_rules(reply, emotion, user_message):

                return reply

            return _safe_fallback(emotion)



    # 3. Rule-based fallback

    return _safe_fallback(emotion)





def generate_response_stream(

    user_message:     str,

    emotion:          str,

    history:          list[dict],

    user_name:        str = "Friend",

    dominant_mood:    Optional[str] = None,

    pattern_insights: list[str] = [],

    weighted_memory:  list[dict] = [],

    transition:       Optional[dict] = None,

    achievements:     list[str] = [],

    past_journals:    list[dict] = [],

) -> Generator[str, None, None]:

    if is_crisis(user_message):

        yield CRISIS_RESPONSE

        return



    # 1. Custom model — yield word by word for streaming effect

    if _custom_model_available():

        if achievements or past_journals:

            context_parts = []

            if achievements:

                context_parts.append("ACHIEVEMENTS: " + " | ".join(achievements))

            if past_journals:

                j_texts = [j.get("content", "") for j in past_journals]

                context_parts.append("JOURNALS: " + " | ".join(j_texts))

            

            user_message = f"{user_message} [{' --- '.join(context_parts)}]"

            

        reply = _call_custom_model(user_message, emotion)

        if reply:

            if _reply_breaks_emotional_rules(reply, emotion, user_message):

                reply = _safe_fallback(emotion)

            for word in reply.split():

                yield word + " "

            return



    # 2. Ollama streaming

    model = get_available_model()

    if model:

        messages = _build_messages(

            user_message, emotion, history, user_name,

            dominant_mood, pattern_insights, weighted_memory, transition, achievements, past_journals

        )

        try:

            chunks = []

            for token in _call_ollama_stream(messages, model):

                chunks.append(token)

            reply = "".join(chunks)

            if _reply_breaks_emotional_rules(reply, emotion, user_message):

                reply = _safe_fallback(emotion)

            yield reply

            return

        except Exception:

            pass



    yield _safe_fallback(emotion)

