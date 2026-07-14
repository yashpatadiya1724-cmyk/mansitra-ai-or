"""
Manasitra Backend — FastAPI v2
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Endpoints:
  GET  /                    — health check
  GET  /ollama/status       — Ollama + model status
  POST /chat                — full blocking chat
  POST /chat/stream         — streaming SSE chat
  GET  /mood/history        — mood history
  GET  /mood/stats          — emotion statistics
  GET  /mood/patterns       — detected patterns (LMPE)
  GET  /mood/memory         — weighted memory (EAMC)
  POST /voice/transcribe    — audio → text
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import sys, os, json, tempfile, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ai-engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ai-engine/custom_model"))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ai-engine"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../ai-engine/custom_model"))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from emotion_model.predictor import predict_emotion
from privacy_engine.privacy import privacy_filter, privacy_metadata
from memory_engine.memory import (
    get_or_create_user,
    save_mood,
    save_message,
    get_mood_history,
    get_recent_conversation,
    get_dominant_mood,
    # Feature 1 — EAMC
    get_weighted_memory,
    # Feature 2 — LMPE
    get_pattern_insights,
    get_mood_patterns,
    # Feature 3 — ESM
    get_last_emotion,
    record_emotion_transition,
    get_current_transition,
    # Feature 4 - Journal
    save_journal_entry,
    get_recent_achievements,
    get_journal_entries,
)
from response_engine.conversation_brain import (
    generate_response,
    generate_response_stream,
    ollama_is_running,
    get_available_model,
)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Manasitra Emotional AI",
    description="Private emotional AI — EAMC + LMPE + ESM",
    version="2.0.0",
)

CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    user_name: str = "Friend"
    message:   str


class ChatResponse(BaseModel):
    reply:       str
    emotion:     str
    confidence:  float
    scores:      dict[str, float]
    emoji:       str = "😐"
    model_used:  str | None = None
    transition:  dict | None = None
    patterns:    list[str] = []
    technical_proof: dict = {}


class JournalRequest(BaseModel):
    user_name:  str = "Friend"
    entry_type: str  # "achievement" or "bad_time"
    content:    str


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "app": "Manasitra Emotional AI v2.0"}


@app.get("/ollama/status")
def ollama_status():
    running = ollama_is_running()
    model   = get_available_model() if running else None
    # Check custom model
    custom_ready = False
    custom_epoch = 0
    try:
        import torch
        ckpt_path = os.path.join(os.path.dirname(__file__), "../ai-engine/custom_model/saved_model/manasitra_model.pt")
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            custom_epoch = ckpt.get("epoch", 0)
            custom_ready = custom_epoch >= 20
    except Exception:
        pass

    active = "custom" if custom_ready else ("ollama" if running and model else "fallback")
    return {
        "ollama_running":  running,
        "model":           model,
        "ready":           running and model is not None,
        "custom_model":    custom_ready,
        "custom_epoch":    custom_epoch,
        "training_progress": f"{custom_epoch}/50 epochs",
        "active_engine":   active,
    }


@app.get("/innovation/status")
def innovation_status():
    """Patent-readiness style technical feature inventory."""
    status = ollama_status()
    return {
        "innovation_claims": {
            "unique_emotion_architecture": True,
            "custom_memory_transition_engine": True,
            "low_latency_offline_response": True,
            "privacy_preserving_local_pipeline": True,
            "measurable_training_inference_method": True,
        },
        "active_engine": status["active_engine"],
        "custom_training_progress": status["training_progress"],
        "external_api_required": False,
        "local_components": [
            "emotion_model.predictor",
            "memory_engine.EAMC_LMPE_ESM",
            "privacy_engine.privacy_filter",
            "response_engine.conversation_brain",
            "custom_model.inference",
        ],
        "evidence_endpoints": [
            "/ollama/status",
            "/mood/memory",
            "/mood/patterns",
            "/innovation/status",
        ],
    }


# ── Chat (blocking) ───────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    started_at = time.perf_counter()
    privacy = privacy_filter(req.message)
    safe_message = privacy.redacted_text

    # 1. Emotion detection
    emotion_result = predict_emotion(safe_message)
    emotion        = emotion_result["emotion"]

    # 2. User + memory
    user_id = get_or_create_user(req.user_name)

    # Feature 3 — ESM: detect transition BEFORE saving new mood
    last_emotion = get_last_emotion(user_id)
    if last_emotion and last_emotion != emotion:
        record_emotion_transition(user_id, last_emotion, emotion)
    transition = get_current_transition(user_id)

    # Save mood + message
    save_mood(user_id, emotion, safe_message)
    save_message(user_id, "user", safe_message, emotion)

    # Feature 1 — EAMC: weighted memory
    weighted_memory = get_weighted_memory(user_id, limit=5)

    # Feature 2 — LMPE: pattern insights
    pattern_insights = get_pattern_insights(user_id)

    # Standard context
    history       = get_recent_conversation(user_id, limit=10)
    dominant_mood = get_dominant_mood(user_id, limit=10)
    model_used    = get_available_model()

    distressed = emotion in ["sad", "lonely", "stress", "anxiety", "angry", "overwhelmed", "grief"]
    achievements = get_recent_achievements(user_id, limit=3) if distressed else []
    past_journals = get_journal_entries(user_id, limit=3) if distressed else []

    # 3. Generate response with all 4 features
    reply = generate_response(
        user_message     = safe_message,
        emotion          = emotion,
        history          = history,
        user_name        = req.user_name,
        dominant_mood    = dominant_mood,
        pattern_insights = pattern_insights,
        weighted_memory  = weighted_memory,
        transition       = transition,
        achievements     = achievements,
        past_journals    = past_journals,
    )

    save_message(user_id, "assistant", reply)
    latency_ms = round((time.perf_counter() - started_at) * 1000, 2)

    return ChatResponse(
        reply       = reply,
        emotion     = emotion,
        confidence  = emotion_result["confidence"],
        scores      = emotion_result["scores"],
        emoji       = emotion_result.get("emoji", "😐"),
        model_used  = model_used,
        transition  = dict(transition) if transition else None,
        patterns    = pattern_insights,
        technical_proof = {
            "latency_ms": latency_ms,
            "engine_route": "custom" if ollama_status()["active_engine"] == "custom" else "ollama_or_fallback",
            "privacy": privacy_metadata(privacy),
            "memory_features_used": {
                "weighted_memory_count": len(weighted_memory),
                "pattern_count": len(pattern_insights),
                "transition_detected": transition is not None,
            },
            "emotion_architecture": "hybrid_keyword_compound_transition",
        },
    )


# ── Chat (streaming SSE) ──────────────────────────────────────────────────────
@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")
    started_at = time.perf_counter()
    privacy = privacy_filter(req.message)
    safe_message = privacy.redacted_text

    emotion_result = predict_emotion(safe_message)
    emotion        = emotion_result["emotion"]
    user_id        = get_or_create_user(req.user_name)

    # Feature 3 — ESM
    last_emotion = get_last_emotion(user_id)
    if last_emotion and last_emotion != emotion:
        record_emotion_transition(user_id, last_emotion, emotion)
    transition = get_current_transition(user_id)

    save_mood(user_id, emotion, safe_message)
    save_message(user_id, "user", safe_message, emotion)

    # Feature 1 + 2
    weighted_memory  = get_weighted_memory(user_id, limit=5)
    pattern_insights = get_pattern_insights(user_id)

    history       = get_recent_conversation(user_id, limit=10)
    dominant_mood = get_dominant_mood(user_id, limit=10)

    distressed = emotion in ["sad", "lonely", "stress", "anxiety", "angry", "overwhelmed", "grief"]
    achievements = get_recent_achievements(user_id, limit=3) if distressed else []
    past_journals = get_journal_entries(user_id, limit=3) if distressed else []

    def event_generator():
        full_reply = []
        for token in generate_response_stream(
            user_message     = safe_message,
            emotion          = emotion,
            history          = history,
            user_name        = req.user_name,
            dominant_mood    = dominant_mood,
            pattern_insights = pattern_insights,
            weighted_memory  = weighted_memory,
            transition       = transition,
            achievements     = achievements,
            past_journals    = past_journals,
        ):
            full_reply.append(token)
            yield f"data: {json.dumps({'token': token})}\n\n"

        save_message(user_id, "assistant", "".join(full_reply))

        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        yield f"data: {json.dumps({'done': True, 'emotion': emotion, 'confidence': emotion_result['confidence'], 'scores': emotion_result['scores'], 'patterns': pattern_insights, 'transition': dict(transition) if transition else None, 'technical_proof': {'latency_ms': latency_ms, 'engine_route': 'custom' if ollama_status()['active_engine'] == 'custom' else 'ollama_or_fallback', 'privacy': privacy_metadata(privacy), 'memory_features_used': {'weighted_memory_count': len(weighted_memory), 'pattern_count': len(pattern_insights), 'transition_detected': transition is not None}, 'emotion_architecture': 'hybrid_keyword_compound_transition'}})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Mood endpoints ────────────────────────────────────────────────────────────
@app.get("/mood/history")
def mood_history(user_name: str, limit: int = 20):
    user_id = get_or_create_user(user_name)
    return {"user": user_name, "history": get_mood_history(user_id, limit)}


@app.get("/mood/stats")
def mood_stats(user_name: str, limit: int = 30):
    user_id = get_or_create_user(user_name)
    history = get_mood_history(user_id, limit)
    counts: dict[str, int] = {}
    for entry in history:
        counts[entry["emotion"]] = counts.get(entry["emotion"], 0) + 1
    total = sum(counts.values()) or 1
    return {
        "user":          user_name,
        "total_entries": len(history),
        "counts":        counts,
        "percentages":   {k: round(v / total * 100, 1) for k, v in counts.items()},
        "dominant_mood": max(counts, key=counts.get) if counts else None,
    }


@app.get("/mood/patterns")
def mood_patterns(user_name: str):
    """Feature 2 — LMPE: returns detected mood patterns."""
    user_id  = get_or_create_user(user_name)
    patterns = get_mood_patterns(user_id)
    insights = get_pattern_insights(user_id)
    return {"user": user_name, "patterns": patterns, "insights": insights}


@app.get("/mood/memory")
def mood_memory(user_name: str, limit: int = 10):
    """Feature 1 — EAMC: returns weighted memory sorted by retention score."""
    user_id = get_or_create_user(user_name)
    memory  = get_weighted_memory(user_id, limit)
    return {"user": user_name, "weighted_memory": memory}


# ── Voice ─────────────────────────────────────────────────────────────────────
@app.post("/voice/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        from voice_engine.voice import transcribe_audio
    except ImportError:
        raise HTTPException(status_code=501, detail="Voice engine not installed.")

    suffix = os.path.splitext(file.filename or "audio.wav")[1] or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name
    try:
        text = transcribe_audio(tmp_path)
    finally:
        os.unlink(tmp_path)
    return {"transcription": text}

# ── Journal endpoints ─────────────────────────────────────────────────────────
@app.post("/journal")
def add_journal_entry(req: JournalRequest):
    user_id = get_or_create_user(req.user_name)
    save_journal_entry(user_id, req.entry_type, req.content)
    return {"status": "ok", "message": "Journal entry saved successfully."}

@app.get("/journal")
def get_journal(user_name: str, limit: int = 20):
    user_id = get_or_create_user(user_name)
    entries = get_journal_entries(user_id, limit)
    return {"user": user_name, "entries": entries}
