import torch
import torch.nn as nn
from Triton_Layers.Seq2Seq import PositionalEncoding, TokenEmbedding
from Triton_Layers.ImageEncoder import CNNBackbone
from Triton_Layers.Encoder import Encoder
from Triton_Layers.Decoder import Decoder
from Triton_Layers.Lif import LIF

class PretrainEncoderModel(nn.Module):
    """
    Modèle pour le pré-entraînement de l'encodeur sur ImageNet.
    Architecture: CNNBackbone (RGB) -> Encoder -> GlobalAvgPool -> Linear(num_classes)
    """
    def __init__(self, 
                 num_classes=1000, 
                 d_model=512, 
                 patch_size=4, 
                 nb_sps_blocks=4, 
                 num_encoder_layers=6, 
                 n_heads=8, 
                 ff_dim=2048, 
                 dropout=0.1,
                 n_steps=10):
        super().__init__()
        self.n_steps = n_steps
        
        # 1. Image Encoder (CNN Backbone) - RGB input (in_channels=3)
        self.image_encoder = CNNBackbone(
            nb_layers=nb_sps_blocks,
            patch_size=patch_size,
            d_model=d_model,
            n_steps=n_steps,
            in_channels=3  # ImageNet est RGB
        )
        
        # 2. Transformer Encoder
        self.encoder = Encoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            n_steps=n_steps
        )
        
        # 3. Classifier Head (ImageNet)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: [B, 3, H, W]
        B, C, H, W = x.shape
        T = self.n_steps
        
        # Expansion temporelle pour le SNN
        x = x.unsqueeze(0).expand(T, B, C, H, W).reshape(T * B, C, H, W)
        
        # Forward backbone
        enc_output, _ = self.image_encoder(x)  # [T*B, N, D]
        
        # Forward encoder
        enc_output = self.encoder(enc_output)  # [T*B, N, D]
        
        # Global Average Pooling sur la dimension spatiale (N)
        # enc_output: [T*B, N, D] -> [T*B, D]
        pooled = enc_output.mean(dim=1)
        
        # Classifier
        logits = self.classifier(pooled)  # [T*B, num_classes]
        
        # Reshape pour séparer steps (optionnel, selon loss)
        logits = logits.view(T, B, -1)   # [T, B, num_classes]
        
        return logits


class PretrainDecoderModel(nn.Module):
    """
    Modèle pour le pré-entraînement du décodeur sur Wikipedia (Causal LM).
    Architecture: TokenEmb -> PosEnc -> Decoder (No Cross-Attn) -> Linear(vocab)
    """
    def __init__(self, 
                 vocab_size=1000, 
                 d_model=512, 
                 num_decoder_layers=6, 
                 n_heads=8, 
                 ff_dim=2048, 
                 dropout=0.1,
                 n_steps=10):
        super().__init__()
        self.n_steps = n_steps
        self.d_model = d_model
        
        self.tgt_tok_emb = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout=dropout)
        self.lifPE = LIF(n_steps=n_steps)
        
        self.decoder = Decoder(
            num_layers=num_decoder_layers,
            d_model=d_model,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
            n_steps=n_steps
        )
        
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, look_ahead_mask=None, enc_output=None):
        # tgt: [B, L] (indices)
        
        # Embedding + PosEncoding
        tgt = self.positional_encoding(self.tgt_tok_emb(tgt))  # [B, L, D]
        
        B, L, D = tgt.shape
        T = self.n_steps
        
        # Expansion temp + LIF
        tgt = tgt.unsqueeze(0).expand(T, B, L, D).reshape(T*B, L, D)
        tgt, _ = self.lifPE(tgt)
        
        # Mask expansion
        if look_ahead_mask is not None:
             look_ahead_mask = look_ahead_mask.unsqueeze(0).repeat(T, 1, 1, 1, 1).reshape(T*B, 1, L, L)

        # Si enc_output est None (ce qui est le cas en pretraining pur décodeur),
        # on doit fournir un tenseur dummy ou modifier le décodeur pour skip cross-attn.
        # Pour éviter de modifier le code du Decoder existant, on passe un enc_output de zéros
        # qui n'aura aucun effet si on est malin, ou alors on accepte que cross-attn
        # apprenne à ignorer ce bruit constant.
        # MIEUX: Modifier DecoderLayer pour accepter enc_output=None et skip cross-attn.
        # Mais ici on veut utiliser le code existant.
        # Option: Passer zeros [T*B, 1, D].
        if enc_output is None:
             enc_output = torch.zeros(T*B, 1, D, device=tgt.device)
        
        # Decoder forward
        # Note: Le decoder va faire self-attn PUIS cross-attn sur des zéros.
        # Ce n'est pas idéal pour un pur LM, mais ça permet de pré-entrainer les poids de self-attn et FFN.
        dec_output = self.decoder(tgt, enc_output, mask=look_ahead_mask) # [T*B, L, D]
        
        logits = self.output_layer(dec_output) # [T*B, L, V]
        logits = logits.view(T, B, L, -1)      # [T, B, L, V]
        
        return logits
