"""
Manasitra Custom Model — Training Script
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Trains the Seq2Seq Transformer on therapy_dataset.json
No GPU required — runs on CPU (takes ~5-15 min)
Output: saved_model/manasitra_model.pt + tokenizer.json
"""

import json
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import ManasitraTransformer, ManasitraTokenizer

# ── Config & Architecture Scaling (Profiles: small, base [146M], large [372M], xlarge [758M]) ──
from model import MODEL_PROFILES, get_model_profile, build_model

PROFILE_NAME = os.getenv("MODEL_PROFILE", "base")
_cfg         = get_model_profile(PROFILE_NAME)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "psychology_dataset.json")
SAVE_DIR     = os.path.join(os.path.dirname(__file__), "saved_model")
MAX_LEN      = _cfg["max_len"]
BATCH_SIZE   = int(os.getenv("BATCH_SIZE", "8" if PROFILE_NAME in ("large", "xlarge") else "16"))
EPOCHS       = int(os.getenv("EPOCHS", "150"))
LR           = float(os.getenv("LR", "1.5e-4" if PROFILE_NAME in ("large", "xlarge") else "2e-4"))
D_MODEL      = _cfg["d_model"]
NHEAD        = _cfg["nhead"]
ENC_LAYERS   = _cfg["enc_layers"]
DEC_LAYERS   = _cfg["dec_layers"]
FFN_DIM      = _cfg["ffn_dim"]
DROPOUT      = _cfg["dropout"]

os.makedirs(SAVE_DIR, exist_ok=True)


# ── Dataset ───────────────────────────────────────────────────────────────────
class TherapyDataset(Dataset):
    def __init__(self, data: list[dict], tokenizer: ManasitraTokenizer, max_len: int):
        self.tokenizer = tokenizer
        self.max_len   = max_len
        self.pairs     = [(d["input"], d["response"]) for d in data]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.pairs[idx]
        src = torch.tensor(self.tokenizer.encode(src_text, self.max_len), dtype=torch.long)
        tgt = torch.tensor(self.tokenizer.encode(tgt_text, self.max_len), dtype=torch.long)
        return src, tgt


# ── Training ──────────────────────────────────────────────────────────────────
def train():
    print("=" * 60)
    print("  Manasitra Custom Model Training")
    print("=" * 60)

    # Load data (auto-generate if missing in fresh clone)
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}. Auto-generating 40,000+ samples...")
        try:
            import generate_massive_dataset
            generate_massive_dataset.generate_pairs()
        except Exception as e:
            print(f"Could not auto-generate dataset: {e}")

    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)

    print(f"Dataset: {len(data)} samples")

    # Build tokenizer
    all_texts = [d["input"] for d in data] + [d["response"] for d in data]
    tokenizer = ManasitraTokenizer()
    tokenizer.build_vocab(all_texts, min_freq=1)
    tokenizer.save(os.path.join(SAVE_DIR, "tokenizer.json"))
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Dataset + DataLoader
    dataset    = TherapyDataset(data, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ManasitraTransformer(
        vocab_size          = tokenizer.vocab_size,
        d_model             = D_MODEL,
        nhead               = NHEAD,
        num_encoder_layers  = ENC_LAYERS,
        num_decoder_layers  = DEC_LAYERS,
        dim_feedforward     = FFN_DIM,
        dropout             = DROPOUT,
        max_len             = MAX_LEN,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss + optimizer
    pad_idx   = tokenizer.word2idx[ManasitraTokenizer.PAD]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_loss = float("inf")
    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    print(f"\nTraining started on {device} (CUDA AMP FP16: {use_cuda})...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            # Teacher forcing: decoder input = tgt[:-1], target = tgt[1:]
            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_pad_mask = (src == pad_idx)
            tgt_pad_mask = (tgt_in == pad_idx)

            with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
                logits = model(src, tgt_in, src_pad_mask, tgt_pad_mask)
                loss = criterion(
                    logits.reshape(-1, tokenizer.vocab_size),
                    tgt_out.reshape(-1),
                )

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        ppl      = math.exp(min(avg_loss, 20))

        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{EPOCHS} | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            cfg_dict = {
                "d_model":    D_MODEL,
                "nhead":      NHEAD,
                "enc_layers": ENC_LAYERS,
                "dec_layers": DEC_LAYERS,
                "ffn_dim":    FFN_DIM,
                "dropout":    DROPOUT,
                "max_len":    MAX_LEN,
            }
            torch.save({
                "epoch":       epoch,
                "profile":     PROFILE_NAME,
                "model_state": model.state_dict(),
                "vocab_size":  tokenizer.vocab_size,
                "config":      cfg_dict,
            }, os.path.join(SAVE_DIR, "manasitra_model.pt"))

            meta = {
                "epoch": epoch,
                "profile": PROFILE_NAME,
                "vocab_size": tokenizer.vocab_size,
                "best_loss": float(best_loss),
                "config": cfg_dict,
            }
            with open(os.path.join(SAVE_DIR, "model_metadata.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

    print(f"\n[DONE] Training complete! Best loss: {best_loss:.4f}")
    print(f"   Model saved to: {SAVE_DIR}/manasitra_model.pt")
    print(f"   Tokenizer saved to: {SAVE_DIR}/tokenizer.json")

    # Quick test
    print("\n--- Quick inference test ---")
    _test_inference(model, tokenizer, device)


def _test_inference(model, tokenizer, device):
    tests = [
        "I feel so lonely, nobody talks to me",
        "I am so stressed about my exams",
        "I feel happy today",
    ]
    bos = tokenizer.word2idx[ManasitraTokenizer.BOS]
    eos = tokenizer.word2idx[ManasitraTokenizer.EOS]
    pad = tokenizer.word2idx[ManasitraTokenizer.PAD]

    for text in tests:
        src = torch.tensor([tokenizer.encode(text, 64)], dtype=torch.long).to(device)
        src_mask = (src == pad)
        ids    = model.generate(src, src_mask, bos, eos, max_len=80, temperature=0.7, top_k=30)
        reply  = tokenizer.decode(ids)
        print(f"  Input : {text}")
        print(f"  Reply : {reply}")
        print()


if __name__ == "__main__":
    train()
