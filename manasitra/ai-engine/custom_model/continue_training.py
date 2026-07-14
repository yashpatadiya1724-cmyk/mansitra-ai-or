"""
Manasitra — Continue Training from Checkpoint
Loads the saved model and trains for more epochs with a lower LR.
"""

import json
import os
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model import ManasitraTransformer, ManasitraTokenizer

# -- Config --
DATASET_PATH = os.path.join(os.path.dirname(__file__), "psychology_dataset.json")
SAVE_DIR     = os.path.join(os.path.dirname(__file__), "saved_model")
MODEL_PATH   = os.path.join(SAVE_DIR, "manasitra_model.pt")
TOK_PATH     = os.path.join(SAVE_DIR, "tokenizer.json")

MAX_LEN      = 64
BATCH_SIZE   = 64
EXTRA_EPOCHS = 50       # train 50 more epochs
LR           = 1e-4     # lower LR for fine-tuning (was 5e-4)
DROPOUT      = 0.15

os.makedirs(SAVE_DIR, exist_ok=True)


# -- Dataset --
class TherapyDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
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


def continue_train():
    print("=" * 60)
    print("  Manasitra — Continue Training")
    print("=" * 60)

    # Load checkpoint
    if not os.path.exists(MODEL_PATH):
        print("No saved model found! Run train.py first.")
        return

    checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    prev_epoch = checkpoint["epoch"]
    print(f"Resuming from epoch {prev_epoch}")
    print(f"Training {EXTRA_EPOCHS} more epochs (LR={LR})")

    # Load tokenizer
    tokenizer = ManasitraTokenizer.load(TOK_PATH)
    print(f"Vocabulary size: {tokenizer.vocab_size}")

    # Load data
    with open(DATASET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    print(f"Dataset: {len(data)} samples")

    dataset    = TherapyDataset(data, tokenizer, MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Rebuild model and load weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ManasitraTransformer(
        vocab_size          = checkpoint["vocab_size"],
        d_model             = cfg["d_model"],
        nhead               = cfg["nhead"],
        num_encoder_layers  = cfg["enc_layers"],
        num_decoder_layers  = cfg["dec_layers"],
        dim_feedforward     = cfg["ffn_dim"],
        dropout             = DROPOUT,
        max_len             = cfg["max_len"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])

    # Loss + optimizer (fresh optimizer with lower LR)
    pad_idx   = tokenizer.word2idx[ManasitraTokenizer.PAD]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx, label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EXTRA_EPOCHS)

    best_loss = float("inf")

    print("\nContinued training started...\n")

    for epoch in range(1, EXTRA_EPOCHS + 1):
        global_epoch = prev_epoch + epoch
        model.train()
        total_loss = 0.0

        for src, tgt in dataloader:
            src, tgt = src.to(device), tgt.to(device)

            tgt_in  = tgt[:, :-1]
            tgt_out = tgt[:, 1:]

            src_pad_mask = (src == pad_idx)
            tgt_pad_mask = (tgt_in == pad_idx)

            logits = model(src, tgt_in, src_pad_mask, tgt_pad_mask)
            loss = criterion(
                logits.reshape(-1, tokenizer.vocab_size),
                tgt_out.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        ppl      = math.exp(min(avg_loss, 20))

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {global_epoch:3d} (+{epoch}) | Loss: {avg_loss:.4f} | PPL: {ppl:.2f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                "epoch":       global_epoch,
                "model_state": model.state_dict(),
                "vocab_size":  tokenizer.vocab_size,
                "config":      cfg,
            }, MODEL_PATH)

    print(f"\n[DONE] Training complete! Best loss: {best_loss:.4f}")
    print(f"   Model saved to: {MODEL_PATH}")

    # Quick test
    print("\n--- Quick inference test ---")
    model.eval()
    bos = tokenizer.word2idx[ManasitraTokenizer.BOS]
    eos = tokenizer.word2idx[ManasitraTokenizer.EOS]
    pad = tokenizer.word2idx[ManasitraTokenizer.PAD]

    tests = [
        "I feel so lonely, nobody talks to me",
        "I am so stressed about my exams",
        "I feel happy today",
        "I am so angry at my friend",
        "I keep worrying about everything",
    ]
    for text in tests:
        src = torch.tensor([tokenizer.encode(text, 64)], dtype=torch.long).to(device)
        src_mask = (src == pad)
        ids   = model.generate(src, src_mask, bos, eos, max_len=80, temperature=0.7, top_k=30)
        reply = tokenizer.decode(ids)
        print(f"  Input : {text}")
        print(f"  Reply : {reply}")
        print()


if __name__ == "__main__":
    continue_train()
