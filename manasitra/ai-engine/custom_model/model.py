"""

Manasitra Custom Neural Model — Built from Scratch

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Architecture: Seq2Seq Transformer (Encoder-Decoder)

No external LLM. No Ollama. Pure PyTorch.



Components:

  - Custom BPE-style tokenizer (character + word level)

  - Positional Encoding

  - Multi-Head Self-Attention

  - Encoder (N layers)

  - Decoder (N layers) with cross-attention

  - Beam Search decoding

"""



import math

import torch

import torch.nn as nn

import torch.nn.functional as F

from typing import Optional





# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# TOKENIZER

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



class ManasitraTokenizer:

    """

    Word-level tokenizer with special tokens.

    Builds vocabulary from training data.

    """

    PAD   = "<PAD>"

    UNK   = "<UNK>"

    BOS   = "<BOS>"

    EOS   = "<EOS>"

    SPECIALS = [PAD, UNK, BOS, EOS]



    def __init__(self):

        self.word2idx: dict[str, int] = {}

        self.idx2word: dict[int, str] = {}

        self.vocab_size = 0



    def build_vocab(self, sentences: list[str], min_freq: int = 1):

        from collections import Counter

        counter = Counter()

        for s in sentences:

            counter.update(self._split(s))



        self.word2idx = {tok: i for i, tok in enumerate(self.SPECIALS)}

        for word, freq in counter.items():

            if freq >= min_freq and word not in self.word2idx:

                self.word2idx[word] = len(self.word2idx)



        self.idx2word  = {i: w for w, i in self.word2idx.items()}

        self.vocab_size = len(self.word2idx)



    def _split(self, text: str) -> list[str]:

        import re

        # Split on whitespace + punctuation but keep punctuation as tokens

        tokens = re.findall(r"\w+|[^\w\s]", text.lower())

        return tokens



    def encode(self, text: str, max_len: int = 64) -> list[int]:

        tokens = [self.BOS] + self._split(text) + [self.EOS]

        ids = [self.word2idx.get(t, self.word2idx[self.UNK]) for t in tokens]

        # Pad or truncate

        if len(ids) < max_len:

            ids += [self.word2idx[self.PAD]] * (max_len - len(ids))

        else:

            ids = ids[:max_len - 1] + [self.word2idx[self.EOS]]

        return ids



    def decode(self, ids: list[int]) -> str:

        words = []

        for i in ids:

            word = self.idx2word.get(i, self.UNK)

            if word in (self.PAD, self.BOS, self.UNK):

                continue

            if word == self.EOS:

                break

            words.append(word)

        # Reconstruct spacing

        text = ""

        for i, w in enumerate(words):

            if i == 0:

                text += w

            elif w in (".", ",", "!", "?", "'", ":", ";", ")", "'s", "n't", "'re", "'ve", "'ll", "'d", "'m"):

                text += w

            elif text and text[-1] == "(":

                text += w

            elif w == "'" and i + 1 < len(words):

                text += w  # apostrophe — no space before

            else:

                text += " " + w

        # Fix common tokenization artifacts

        import re

        text = re.sub(r"\s+'", "'", text)       # space before apostrophe

        text = re.sub(r"'\s+s\b", "'s", text)   # it ' s → it's

        text = re.sub(r"'\s+re\b", "'re", text)

        text = re.sub(r"'\s+ve\b", "'ve", text)

        text = re.sub(r"'\s+ll\b", "'ll", text)

        text = re.sub(r"'\s+t\b",  "'t",  text)

        text = re.sub(r"'\s+m\b",  "'m",  text)

        text = re.sub(r"'\s+d\b",  "'d",  text)

        # Fix â encoding artifact

        text = text.replace("â", "—").replace("â€™", "'").replace("â€\"", "—")

        text = re.sub(r'\s+', ' ', text)

        # Capitalize first letter

        text = text.strip()

        if text:

            text = text[0].upper() + text[1:]

        return text



    def save(self, path: str):

        import json

        with open(path, "w") as f:

            json.dump({"word2idx": self.word2idx}, f)



    @classmethod

    def load(cls, path: str) -> "ManasitraTokenizer":

        import json

        tok = cls()

        with open(path) as f:

            data = json.load(f)

        tok.word2idx  = data["word2idx"]

        tok.idx2word  = {int(i): w for w, i in tok.word2idx.items()}

        tok.vocab_size = len(tok.word2idx)

        return tok





# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# POSITIONAL ENCODING

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):

        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        pos = torch.arange(0, max_len).unsqueeze(1).float()

        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(pos * div)

        pe[:, 1::2] = torch.cos(pos * div)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)



    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = x + self.pe[:, :x.size(1)]

        return self.dropout(x)





# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# TRANSFORMER MODEL

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━



class ManasitraTransformer(nn.Module):

    """

    Seq2Seq Transformer — Encoder + Decoder.

    Trained entirely on therapy conversation data.

    """



    def __init__(

        self,

        vocab_size:  int,

        d_model:     int = 256,

        nhead:       int = 8,

        num_encoder_layers: int = 4,

        num_decoder_layers: int = 4,

        dim_feedforward: int = 512,

        dropout:     float = 0.1,

        max_len:     int = 128,

    ):

        super().__init__()

        self.d_model   = d_model

        self.max_len   = max_len



        # Embeddings

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)

        self.pos_enc   = PositionalEncoding(d_model, max_len, dropout)



        # Transformer core

        encoder_layer = nn.TransformerEncoderLayer(

            d_model, nhead, dim_feedforward, dropout, batch_first=True

        )

        decoder_layer = nn.TransformerDecoderLayer(

            d_model, nhead, dim_feedforward, dropout, batch_first=True

        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)

        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)



        # Output projection

        self.fc_out = nn.Linear(d_model, vocab_size)



        self._init_weights()



    def _init_weights(self):

        for p in self.parameters():

            if p.dim() > 1:

                nn.init.xavier_uniform_(p)



    def encode(self, src: torch.Tensor, src_key_padding_mask: torch.Tensor) -> torch.Tensor:

        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))

        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)



    def decode(

        self,

        tgt: torch.Tensor,

        memory: torch.Tensor,

        tgt_mask: torch.Tensor,

        tgt_key_padding_mask: torch.Tensor,

        memory_key_padding_mask: torch.Tensor,

    ) -> torch.Tensor:

        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        out = self.decoder(

            x, memory,

            tgt_mask=tgt_mask,

            tgt_key_padding_mask=tgt_key_padding_mask,

            memory_key_padding_mask=memory_key_padding_mask,

        )

        return self.fc_out(out)



    def forward(

        self,

        src: torch.Tensor,

        tgt: torch.Tensor,

        src_key_padding_mask: torch.Tensor,

        tgt_key_padding_mask: torch.Tensor,

    ) -> torch.Tensor:

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(

            tgt.size(1), device=src.device

        ).bool()

        memory = self.encode(src, src_key_padding_mask)

        return self.decode(tgt, memory, tgt_mask, tgt_key_padding_mask, src_key_padding_mask)



    @torch.no_grad()
    @torch.inference_mode()
    def generate(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
        bos_idx: int,
        eos_idx: int,
        max_len: int = 50,
        temperature: float = 0.8,
        top_k: int = 30,
        repetition_penalty: float = 1.25,
        early_stop_ids: tuple = (),
    ) -> list[int]:
        """
        High-performance generation with torch.inference_mode, preallocated causal mask,
        special token suppression, repetition penalty, and adaptive early sentence termination.
        """
        self.eval()
        memory = self.encode(src, src_key_padding_mask)
        ys = torch.tensor([[bos_idx]], device=src.device)
        suppress_tokens = {0, 1, bos_idx}

        # Preallocate causal mask view instead of recreating every token step
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            max_len + 2, device=src.device
        ).bool()

        stop_set = {eos_idx} | set(early_stop_ids)
        min_sentence_len = 10

        for _ in range(max_len):
            cur_len = ys.size(1)
            tgt_mask = causal_mask[:cur_len, :cur_len]
            tgt_pad_mask = (ys == 0)
            out = self.decode(ys, memory, tgt_mask, tgt_pad_mask, src_key_padding_mask)
            logits = out[:, -1, :].clone() / max(temperature, 1e-5)

            # Suppress special / unknown tokens from generation
            for tok in suppress_tokens:
                logits[:, tok] = float("-inf")

            # Repetition penalty for already emitted tokens
            if repetition_penalty > 1.0:
                for token_id in set(ys[0].tolist()):
                    if token_id not in (0, 1, bos_idx, eos_idx):
                        if logits[:, token_id] > 0:
                            logits[:, token_id] /= repetition_penalty
                        else:
                            logits[:, token_id] *= repetition_penalty

            # Top-k filtering
            if top_k > 0:
                k = min(top_k, logits.size(-1))
                values, _ = torch.topk(logits, k)
                min_val = values[:, -1].unsqueeze(-1)
                logits = logits.masked_fill(logits < min_val, float("-inf"))

            probs     = F.softmax(logits, dim=-1)
            next_tok  = torch.multinomial(probs, 1)
            tok_id    = next_tok.item()
            ys        = torch.cat([ys, next_tok], dim=1)

            if tok_id == eos_idx:
                break
            if cur_len >= min_sentence_len and tok_id in stop_set:
                break

        return ys[0, 1:].tolist()  # skip BOS
