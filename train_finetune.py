
import os
import torch
import torch.nn as nn
from models_pretraining import PretrainEncoderModel, PretrainDecoderModel
from Triton_Layers.Seq2Seq import Seq2Seq
from train_spikeformer import train as run_training # Reuse existing logic with modifications?
# Actually cleaner to rewrite main loop to load weights and freeze specific parts.

from transformers import TrOCRProcessor
from torch.utils.data import DataLoader
from wiki_text_images3 import WikiTextImageDataset, WikiTextDataCollator
from train_spikeformer import train
# Wait, train_spikeformer.py instantiates a fresh Seq2Seq.
# We need to modify train_spikeformer.py OR create a new one that loads the model differently.
# Let's create a new one to keep things clean.

import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from Triton_Layers.Seq2Seq import Seq2Seq as Seq2SeqTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_pretrained_weights(full_model, enc_path, dec_path):
    print(f"Loading pretrained weights...")
    
    # Encoder
    if os.path.exists(enc_path):
        enc_ckpt = torch.load(enc_path, map_location="cpu") # Load to CPU first
        # Keys in enc_ckpt: 'image_encoder.xxx', 'encoder.xxx', 'classifier.xxx'
        # full_model keys: 'image_encoder.xxx', 'encoder.xxx'
        
        enc_state = {}
        for k, v in enc_ckpt.items():
            if k.startswith("image_encoder.") or k.startswith("encoder."):
                enc_state[k] = v
        
        msg = full_model.load_state_dict(enc_state, strict=False)
        print(f"Encoder weights loaded: {msg}")
    else:
        print(f"⚠️ Encoder checkpoint not found at {enc_path}")

    # Decoder
    if os.path.exists(dec_path):
        dec_ckpt = torch.load(dec_path, map_location="cpu")
        # Keys: 'tgt_tok_emb', 'positional_encoding', 'lifPE', 'decoder.layers.X.self_attns', 'decoder.layers.X.mlp', 'output_layer'
        # full_model keys: 'tgt_tok_emb', 'positional_encoding', 'lifPE', 'decoder.layers.X...'
        
        # We want to transfer self_attns and MLPs, but NOT cross_attns (which are new)
        dec_state = {}
        for k, v in dec_ckpt.items():
            # Skip output layer (or keep it?) - Keep it as it is same vocab
            if "cross_attns" not in k:
                dec_state[k] = v
        
        msg = full_model.load_state_dict(dec_state, strict=False)
        print(f"Decoder weights loaded (excluding cross-attn): {msg}")
    else:
        print(f"⚠️ Decoder checkpoint not found at {dec_path}")

def freeze_layers(model):
    print("Freezing ImageEncoder, Encoder, and Decoder Self-Attn/MLP...")
    
    # Freeze Encoder side
    for param in model.image_encoder.parameters():
        param.requires_grad = False
    for param in model.encoder.parameters():
        param.requires_grad = False
        
    # Freeze Decoder partial
    model.tgt_tok_emb.requires_grad_(False)
    model.positional_encoding.requires_grad_(False)
    model.lifPE.requires_grad_(False)
    
    for layer in model.decoder.layers:
        # Freeze Self-Attn and MLP
        for param in layer.self_attns.parameters():
            param.requires_grad = False
        for param in layer.mlp.parameters():
            param.requires_grad = False
        
        # Keep Cross-Attn trainable (enable grad)
        for param in layer.cross_attns.parameters():
            param.requires_grad = True
            
    print("✅ Layers frozen. Only Cross-Attention trainable.")

def main():
    # ... reused configs ...
    BATCH_SIZE = 96
    LR = 1e-4 # Lower LR for finetuning
    
    # 1. Instantiate Full Model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    model = Seq2SeqTransformer(
        num_encoder_layers=6,
        num_decoder_layers=6,
        d_model=512,
        n_heads=8,
        tgt_vocab_size=len(processor.tokenizer),
        ff_dim=2048,
        n_steps=10,
        in_channels=3 # RGB now
    ).to(DEVICE)
    
    # 2. Load Weights
    load_pretrained_weights(
        model, 
        "checkpoints/encoder_pretrain_best.pt",
        "checkpoints/decoder_pretrain_best.pt"
    )
    
    # 3. Freeze
    freeze_layers(model)
    
    # 4. Train Loop (simplified copy from train_spikeformer.py)
    # Reuse dataset ...
    train_ds = WikiTextImageDataset(
        processor=processor, split="train", img_size=(32, 512), train=True, max_samples=100000 
    )
    collate = WikiTextDataCollator(processor, max_length=128)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=8, collate_fn=collate, persistent_workers=True)
    
    # Optimizer - Only trainable params
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    criterion = nn.CrossEntropyLoss(ignore_index=processor.tokenizer.pad_token_id)
    scaler = GradScaler()
    
    print("Starting Phase 2: Finetuning Cross-Attention...")
    
    for epoch in range(5):
        model.train()
        pbar = tqdm(train_loader)
        for batch in pbar:
            src = batch["pixel_values"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)
            
            # RGB conversion if dataset is grayscale?
            if src.shape[1] == 1:
                src = src.repeat(1, 3, 1, 1)
                
            tgt_in = labels[:, :-1]
            tgt_out = labels[:, 1:]
            
            # MASKS
            look_ahead = model.create_look_ahead_mask(tgt_in.size(1), DEVICE)
            pad_mask = model.create_padding_mask(tgt_in, processor.tokenizer.pad_token_id, DEVICE)
            mask = torch.minimum(pad_mask, look_ahead)
            
            optimizer.zero_grad()
            with autocast():
                logits, _ = model(src, tgt_in, look_ahead_mask=mask)
                
                loss = 0
                for t in range(logits.shape[0]):
                    loss += criterion(logits[t].reshape(-1, len(processor.tokenizer)), tgt_out.reshape(-1))
                loss /= logits.shape[0]
                
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
    print("Finetuning Complete.")

if __name__ == "__main__":
    main()
