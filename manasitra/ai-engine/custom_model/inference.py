"""

Manasitra Custom Model — Inference Engine

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Loads the trained model and generates responses.

No Ollama. No external LLM. 100% custom.

"""



import os

import torch

from functools import lru_cache

from model import ManasitraTransformer, ManasitraTokenizer



SAVE_DIR   = os.path.join(os.path.dirname(__file__), "saved_model")

MODEL_PATH = os.path.join(SAVE_DIR, "manasitra_model.pt")

TOK_PATH   = os.path.join(SAVE_DIR, "tokenizer.json")



# Emotion-aware temperature — distress = more careful/warm, happy = more expressive

EMOTION_TEMP = {

    "sad":     0.65,

    "lonely":  0.65,

    "anxiety": 0.68,

    "stress":  0.70,

    "overwhelmed": 0.66,

    "angry":   0.70,

    "happy":   0.80,

    "neutral": 0.75,

}





@lru_cache(maxsize=1)

def _load() -> tuple:

    """Load model + tokenizer once, cache forever."""

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(

            f"Model not found at {MODEL_PATH}. "

            "Please run: python train.py"

        )



    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)

    cfg        = checkpoint["config"]

    tokenizer  = ManasitraTokenizer.load(TOK_PATH)



    model = ManasitraTransformer(

        vocab_size          = checkpoint["vocab_size"],

        d_model             = cfg["d_model"],

        nhead               = cfg["nhead"],

        num_encoder_layers  = cfg["enc_layers"],

        num_decoder_layers  = cfg["dec_layers"],

        dim_feedforward     = cfg["ffn_dim"],

        dropout             = 0.0,   # no dropout at inference

        max_len             = cfg["max_len"],

    )

    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    return model, tokenizer





def is_model_trained() -> bool:

    return os.path.exists(MODEL_PATH) and os.path.exists(TOK_PATH)





def generate(
    text:        str,
    emotion:     str = "neutral",
    max_len:     int = 50,
    top_k:       int = 30,
) -> str:
    """
    Generate a psychologist response for the given input text with accelerated inference.
    Returns the response string.
    """
    model, tokenizer = _load()

    pad_idx  = tokenizer.word2idx[ManasitraTokenizer.PAD]
    bos_idx  = tokenizer.word2idx[ManasitraTokenizer.BOS]
    eos_idx  = tokenizer.word2idx[ManasitraTokenizer.EOS]
    temp     = EMOTION_TEMP.get(emotion, 0.75)

    early_stop_ids = tuple(
        tokenizer.word2idx[tok] for tok in (".", "?", "!")
        if tok in tokenizer.word2idx
    )

    src      = torch.tensor([tokenizer.encode(text, 64)], dtype=torch.long)
    src_mask = (src == pad_idx)

    with torch.inference_mode():
        ids = model.generate(
            src,
            src_mask,
            bos_idx,
            eos_idx,
            max_len=max_len,
            temperature=temp,
            top_k=top_k,
            early_stop_ids=early_stop_ids,
        )
    reply = tokenizer.decode(ids)
    if not _is_coherent(reply):
        return _fallback(emotion)
    return reply





def _is_coherent(reply: str) -> bool:

    """Verify that generated text has grammatical coherence and minimal lexical quality."""

    import re

    if not reply or len(reply.strip()) < 12:

        return False

    words = re.findall(r"\b[A-Za-z]{2,}\b", reply)

    if len(words) < 4:

        return False

    punct_count = len(re.findall(r"[?.!,;:\-]", reply))

    if punct_count > len(words):

        return False

    return True





def _fallback(emotion: str) -> str:

    import random

    fallbacks = {

        "sad":     "I hear you, and I'm here with you. Can you tell me more about what's going on?",

        "lonely":  "I'm here, and I'm glad you reached out. You're not alone. What's been happening?",

        "anxiety": "You're safe right now. Take one slow breath. What's worrying you most?",

        "stress":  "That's a lot to carry. Let's take it one step at a time. What's most urgent?",

        "overwhelmed": "I'm sorry you're carrying so much right now. What part feels heaviest: studies, expectations, or comparing yourself with others?",

        "angry":   "Your feelings are valid. I'm listening. What happened?",

        "happy":   "That's wonderful! Tell me more — what's making you feel this way?",

        "neutral": "I'm here whenever you need me. What's on your mind?",

    }

    return fallbacks.get(emotion, "I'm here for you. Tell me what's going on.")





if __name__ == "__main__":

    if not is_model_trained():

        print("Model not trained yet. Run: python train.py")

    else:

        tests = [

            ("I feel so lonely, nobody talks to me", "lonely"),

            ("I am so stressed about my exams", "stress"),

            ("I feel happy today, I got good marks", "happy"),

            ("I am so angry at my friend", "angry"),

        ]

        for text, emotion in tests:

            reply = generate(text, emotion)

            print(f"[{emotion}] {text}")

            print(f"  -> {reply}\n")

