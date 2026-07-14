"""Quick model test — run inference on the saved model."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))

from inference import generate, is_model_trained

if not is_model_trained():
    print("Model not trained yet!")
    sys.exit(1)

print("=" * 60)
print("  Manasitra Model — Live Inference Test")
print("=" * 60)

# Check model info
SAVE_DIR = os.path.join(os.path.dirname(__file__), "saved_model")
ckpt = torch.load(os.path.join(SAVE_DIR, "manasitra_model.pt"), map_location="cpu", weights_only=False)
cfg = ckpt["config"]
total_params = sum(
    p.numel() for p in __import__("model").ManasitraTransformer(
        vocab_size=ckpt["vocab_size"],
        d_model=cfg["d_model"],
        nhead=cfg["nhead"],
        num_encoder_layers=cfg["enc_layers"],
        num_decoder_layers=cfg["dec_layers"],
        dim_feedforward=cfg["ffn_dim"],
        max_len=cfg["max_len"],
    ).parameters()
)
print(f"  Parameters : {total_params:,}")
print(f"  Vocab Size : {ckpt['vocab_size']}")
print(f"  Epoch      : {ckpt.get('epoch', '?')}")
print(f"  Round      : {ckpt.get('round', '?')}")
print(f"  Best Loss  : {ckpt.get('best_loss', '?')}")
print(f"  Best Acc   : {ckpt.get('best_accuracy', '?')}")
model_mb = os.path.getsize(os.path.join(SAVE_DIR, "manasitra_model.pt")) / (1024*1024)
print(f"  Model Size : {model_mb:.1f} MB")
print()

tests = [
    ("I feel so lonely, nobody talks to me", "lonely"),
    ("I am so stressed about my exams", "stress"),
    ("I feel happy today, I got good marks", "happy"),
    ("I am so angry at my friend, they betrayed me", "angry"),
    ("I keep worrying about everything", "anxiety"),
    ("I feel worthless and empty inside", "sad"),
    ("I don't see the point in anything anymore", "sad"),
    ("My parents are always fighting", "stress"),
]

print("-" * 60)
print("  Conversations")
print("-" * 60)
for text, emotion in tests:
    reply = generate(text, emotion, max_len=60, top_k=20)
    print(f"  [{emotion:8s}] {text}")
    print(f"           -> {reply}")
    print()
