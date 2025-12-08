import torch.nn as nn
import torch
from .Encoder import Encoder
from .Decoder import Decoder
from .ImageEncoder import CNNBackbone
from .Lif import LIF
import math

from torch import Tensor

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super().__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)  # (1, maxlen, emb_size)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        # token_embedding: (B, T, E)
        return self.dropout(token_embedding + self.pos_embedding[:, : token_embedding.size(1)])


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class Seq2Seq(nn.Module):
    """
    Sequence-to-Sequence model with Spiking Encoder and Decoder.
    Accepts inputs of shape [T, B, C, H, W] and outputs shape [T, B, N, D].
    """
    def __init__(self,
                 patch_size=16,
                 d_model=512,
                 n_heads=8,
                 ff_dim=2048,
                 num_encoder_layers=6,
                 num_decoder_layers=6,
                 dropout=0.1,
                 alpha=0.125,
                 tgt_vocab_size=1000,
                 nb_sps_blocks=4,
                 n_steps=10):
        super().__init__()
        self.n_steps = n_steps
        
        self.image_encoder = CNNBackbone(
            nb_layers=nb_sps_blocks,
            patch_size=patch_size,
            d_model=d_model,
            n_steps=n_steps
        )
        
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            alpha=alpha,
            n_steps=n_steps
        )
        
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            alpha=alpha,
            n_steps=n_steps
        )
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)  # Example output layer
        self.lifPE = LIF(n_steps=n_steps)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)

    def forward(self, src, tgt, look_ahead_mask: Tensor = None, enc_padding_mask: Tensor = None, dec_padding_mask: Tensor = None):
        # src: [B, C, H, W]
        # tgt: [B, N, D]
        enc_output, _ = self.encode(src)  # [T*B, N, D]
        output = self.decode(tgt, enc_output, look_ahead_mask, enc_padding_mask, dec_padding_mask)  # [T*B, N, D] -> [T, B, N, D]
        # output = self.decode(tgt, tgt)  # [T*B, N, D] -> [T, B, N, D]
        return output

    def encode(self, src: Tensor):
        B, C, H, W = src.shape
        T = self.n_steps
        src = src.unsqueeze(0).expand(T, B, C, H, W).reshape(T * B, C, H, W)  # [T*B, C, H, W]
        enc_output, _ = self.image_encoder(src)  # [T*B, N, D]
        enc_output = self.encoder(enc_output)  # [T*B, N, D]
        return enc_output, None

    def decode(self, tgt: Tensor, enc_output: Tensor, look_ahead_mask: Tensor = None, enc_padding_mask: Tensor = None, dec_padding_mask: Tensor = None):
        if look_ahead_mask is not None:
            look_ahead_mask = look_ahead_mask.unsqueeze(0).repeat(self.n_steps, 1, 1, 1, 1).reshape(self.n_steps*look_ahead_mask.shape[0], look_ahead_mask.shape[1], look_ahead_mask.shape[2], look_ahead_mask.shape[3])  # [T*B, 1, N, N]
        tgt = self.positional_encoding(self.tgt_tok_emb(tgt))
        # look_ahead_mask= None
        B, N, D = tgt.shape
        T = self.n_steps
        tgt = tgt.unsqueeze(0).expand(T, B, N, D).reshape(T*B, N, D)  # [T*B, N, D]
        tgt, _ = self.lifPE(tgt)  # Apply LIF to positional encoded target
        
        dec_output = self.decoder(tgt, enc_output, mask=look_ahead_mask)  # [T*B, N, D]
        output = self.output_layer(dec_output)  # [T*B, N, vocab_size]
        output = output.view(self.n_steps, output.shape[0] // self.n_steps, output.shape[1], output.shape[2])
        return output, None  # [T, B, N, vocab_size]
    
    def create_padding_mask(self, seq: Tensor, pad_idx: int, device) -> Tensor:
        # 1 pour garder, 0 pour pad
        return torch.where(seq == pad_idx, 0.0, 1.0).unsqueeze(1).unsqueeze(1).to(device)  # (B,1,1,T)


    def create_look_ahead_mask(self, size: int, device) -> Tensor:
        return torch.tril(torch.ones((size, size), device=device))

    def create_decoder_mask(self, tgt: Tensor, pad_idx: int, device) -> Tensor:
        # Combine padding mask and look-ahead mask: (B, 1, T, T)
        look_ahead = self.create_look_ahead_mask(tgt.shape[1], device)  # (T, T)
        dec_target_padding = self.create_padding_mask(tgt, pad_idx, device)  # (B, 1, 1, T)
        # broadcast to (B,1,T,T)
        return torch.minimum(dec_target_padding, look_ahead)  # float 0/1

    @torch.no_grad()
    def greedy_decode(self, src: Tensor, max_len: int, start_symbol: int, eos_idx: int, pad_idx: int, device: str = "cuda"):
        # src: (C,H,W) or (1,C,H,W)
        if src.dim() == 3:
            src = src.unsqueeze(0)
        src = src.to(device)
        memory, attention_rec = self.encode(src)  # memory: (steps, 1, S, E)

        ys = torch.full((1, 1), fill_value=start_symbol, dtype=torch.long, device=device)  # (1,1)
        dec_attn = None
        out = None

        for _ in range(max_len - 1):
            tgt_mask = self.create_decoder_mask(ys, pad_idx, device)  # (1,1,T,T) float 0/1
            out, dec_attn = self.decode(ys, memory, tgt_mask)    # out: (steps, 1, T, V)

            # Moyenne sur les steps SNN
            prob = out.mean(dim=0)   # (1, T, V)
            next_token_id = torch.argmax(prob[:, -1, :], dim=-1).item()
            ys = torch.cat([ys, torch.tensor([[next_token_id]], device=device, dtype=torch.long)], dim=1)
            if next_token_id == eos_idx:
                break

        # Attentions du dernier decode
        if dec_attn is not None:
            for i in range(len(attention_rec)):
                attention_rec[i]["decoder_attention"] = dec_attn[i]["decoder_attention"]
        return ys, attention_rec, out