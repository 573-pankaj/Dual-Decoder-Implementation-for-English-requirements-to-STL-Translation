"""
Dual-Channel Transformer Decoder
- Two parallel paths: SYNTAX and SEMANTICS
- Each path has its own {Self-Attn → Enc-Dec Attn → FFN} stack
- Per-layer outputs are fused with a learnable gate
- Trainer can supervise:
    L = L_fused + α * L_semantic + β * L_syntax
  by applying CE on:
    - fused logits (always)
    - per-layer semantic states (via decoder.dense)
    - per-layer syntax states (via decoder.dense)
"""

import math
import torch
from torch import nn
from d2l import torch as d2l


class DualDecoderBlock(nn.Module):
    """
    One layer of the dual transformer decoder.
    Each path (syntax / semantics) has its own attention stack.
    Outputs are fused via a learnable gate.
    """
    def __init__(self, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, dropout, layer_id, **kwargs):
        super().__init__(**kwargs)
        self.layer_id = layer_id

        # --- Self-Attention (per path) ---
        self.syn_self_attn = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.sem_self_attn = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )

        # --- Encoder-Decoder Attention (per path) ---
        self.syn_encdec_attn = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )
        self.sem_encdec_attn = d2l.MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout
        )

        # --- FFNs (per path) ---
        self.syn_ffn = d2l.PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.sem_ffn = d2l.PositionWiseFFN(ffn_num_input, ffn_num_hiddens, num_hiddens)

        # --- Add & Norm (per path) ---
        self.addnorm_syn_1 = d2l.AddNorm(norm_shape, dropout)
        self.addnorm_syn_2 = d2l.AddNorm(norm_shape, dropout)
        self.addnorm_syn_3 = d2l.AddNorm(norm_shape, dropout)

        self.addnorm_sem_1 = d2l.AddNorm(norm_shape, dropout)
        self.addnorm_sem_2 = d2l.AddNorm(norm_shape, dropout)
        self.addnorm_sem_3 = d2l.AddNorm(norm_shape, dropout)

        # --- Gated Fusion of both paths ---
        self.fuse_gate = nn.Linear(num_hiddens * 2, num_hiddens)

    def forward(self, X, state):
        """
        Inputs:
            X:          [B, T, H] decoder hidden states from previous layer (or embeddings for layer 0)
            state:      [enc_outputs, enc_valid_lens, cache_syn, cache_sem, Y_valid_len]
        Returns:
            fused:      [B, T, H] fused hidden states of this layer
            Z_syn:      [B, T, H] syntax path output (pre-fusion)
            Z_sem:      [B, T, H] semantic path output (pre-fusion)
            new_state:  updated state
        """
        enc_outputs, enc_valid_lens, cache_syn, cache_sem, Y_valid_len = state

        # Cached decoding (for inference/beam search). During training, we pass full X each step.
        if cache_syn[self.layer_id] is None:
            key_values_syn, key_values_sem = X, X
        else:
            key_values_syn = torch.cat((cache_syn[self.layer_id], X), dim=1)
            key_values_sem = torch.cat((cache_sem[self.layer_id], X), dim=1)
        cache_syn[self.layer_id] = key_values_syn
        cache_sem[self.layer_id] = key_values_sem

        # Subsequent mask lengths for decoder self-attn (causal), only used in training
        if self.training:
            batch_size, num_steps, _ = X.shape
            # shape: [B, T] with [1, 2, ..., T]
            dec_valid_lens = torch.arange(1, num_steps + 1, device=X.device).repeat(batch_size, 1)
        else:
            dec_valid_lens = None

        # -------- SYNTAX path --------
        X_syn = self.syn_self_attn(X, key_values_syn, key_values_syn, dec_valid_lens)   # self-attn
        X_syn = self.addnorm_syn_1(X, X_syn)
        Y_syn = self.syn_encdec_attn(X_syn, enc_outputs, enc_outputs, enc_valid_lens)   # enc-dec attn
        Y_syn = self.addnorm_syn_2(X_syn, Y_syn)
        Z_syn = self.addnorm_syn_3(Y_syn, self.syn_ffn(Y_syn))                          # ffn

        # -------- SEMANTIC path --------
        X_sem = self.sem_self_attn(X, key_values_sem, key_values_sem, dec_valid_lens)
        X_sem = self.addnorm_sem_1(X, X_sem)
        Y_sem = self.sem_encdec_attn(X_sem, enc_outputs, enc_outputs, enc_valid_lens)
        Y_sem = self.addnorm_sem_2(X_sem, Y_sem)
        Z_sem = self.addnorm_sem_3(Y_sem, self.sem_ffn(Y_sem))

        # -------- Gated Fusion --------
        gate = torch.sigmoid(self.fuse_gate(torch.cat((Z_syn, Z_sem), dim=-1)))
        fused = gate * Z_syn + (1.0 - gate) * Z_sem

        # Diagnostic: log gate stats occasionally
        # if not self.training and torch.rand(1).item() < 0.1:  # adjust as needed
        if False:  # adjust as needed
            print(f"[Diag] Layer {self.layer_id} gate mean={gate.mean().item():.3f}, std={gate.std().item():.3f}")

        return fused, Z_syn, Z_sem, [enc_outputs, enc_valid_lens, cache_syn, cache_sem, Y_valid_len]


class DualTransformerDecoder(d2l.AttentionDecoder):
    """
    Transformer decoder with separate syntax and semantic channels.
    forward() returns:
        fused_logits, syn_outputs_per_layer, sem_outputs_per_layer, state
    so the trainer can compute:
        L = L_fused + α * L_semantic + β * L_syntax
    """
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super().__init__(**kwargs)
        self.num_layers = num_layers
        self.num_hiddens = num_hiddens

        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)

        # Stack of dual decoder blocks
        self.blks = nn.ModuleList([
            DualDecoderBlock(key_size, query_size, value_size,
                             num_hiddens, norm_shape, ffn_num_input,
                             ffn_num_hiddens, num_heads, dropout, i)
            for i in range(num_layers)
        ])

        # Final projection to vocab
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, Y_valid_len=None, *args):
        """
        Returns decoder state:
            [enc_outputs, enc_valid_lens, cache_syn, cache_sem, Y_valid_len]
        where cache_* are per-layer caches for fast autoregressive decoding.
        """
        return [
            enc_outputs,
            enc_valid_lens,
            [None] * self.num_layers,  # syntax caches
            [None] * self.num_layers,  # semantic caches
            Y_valid_len
        ]

    def forward(self, X, state):
        """
        Inputs:
            X:      [B, T] token ids (already teacher-forced input in training)
            state:  decoder state from init_state()
        Returns:
            fused_logits:  [B, T, V]
            syn_outputs:   list of per-layer hidden states [B, T, H]
            sem_outputs:   list of per-layer hidden states [B, T, H]
            state:         updated decoder state
        """
        # Embed + position encode
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))

        syn_outputs, sem_outputs = [], []
        for blk in self.blks:
            X, Z_syn, Z_sem, state = blk(X, state)
            syn_outputs.append(Z_syn)
            sem_outputs.append(Z_sem)

        fused_logits = self.dense(X)  # project fused states to vocab
        return fused_logits, syn_outputs, sem_outputs, state
