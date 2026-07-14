import os, torch, sys
sys.path.insert(0, os.path.dirname(__file__))

SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
ckpt = torch.load(os.path.join(SAVE_DIR, "manasitra_model.pt"), map_location="cpu", weights_only=False)

epoch = ckpt["epoch"]
vocab = ckpt["vocab_size"]
cfg = ckpt["config"]

print(f"Epoch: {epoch}")
print(f"Vocab size: {vocab}")
print(f"Config: {cfg}")

# Quick inference test
from inference import generate
tests = [
    ("I feel so lonely, nobody talks to me", "lonely"),
    ("I am so stressed about my exams", "stress"),
    ("I feel happy today", "happy"),
    ("I am so angry at my friend", "angry"),
]
print("\n--- Inference Test ---")
for text, emotion in tests:
    reply = generate(text, emotion, max_len=60, top_k=20)
    print(f"[{emotion}] {text}")
    print(f"  -> {reply}")
    print()
