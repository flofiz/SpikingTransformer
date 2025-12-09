import os
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor
from torch.cuda.amp import autocast, GradScaler
from models_pretraining import PretrainDecoderModel
from wiki_text_images3 import WikiTextDataCollator, MultiSourceTextDataset

# CONFIG
DEVICE = "cuda:1"
BATCH_SIZE = 64
NUM_EPOCHS = 5
LR = 5e-4
MAX_CHARS = 128
NUM_WORKERS = 8
GRAD_CLIP = 1.0

def create_look_ahead_mask(size, device):
    return torch.tril(torch.ones((size, size), device=device))

def main():
    print(f"Starting Decoder Pretraining on {DEVICE}")
    
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    VOCAB_SIZE = len(processor.tokenizer)
    PAD_IDX = processor.tokenizer.pad_token_id
    
    # Dataset (Text Only - we ignore images here)
    # Using existing Dataset class but we only care about text labels
    train_ds = MultiSourceTextDataset(
        processor=processor,
        split="train",
        train=True,
        img_size=(32, 512), # Dummy
        max_chars=MAX_CHARS,
    )
    
    collate = WikiTextDataCollator(processor, max_length=MAX_CHARS)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate, persistent_workers=True)
    
    model = PretrainDecoderModel(vocab_size=VOCAB_SIZE).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    
    steps = 0
    MAX_STEPS = 100000 # Define explicit steps since dataset is infinite
    
    model.train()
    pbar = tqdm(total=MAX_STEPS, desc=f"[Dec GPU:1]")
    
    train_iter = iter(train_loader)
    
    while steps < MAX_STEPS:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        labels = batch["labels"].to(DEVICE) # [B, L]
        
        tgt_in = labels[:, :-1]
        tgt_out = labels[:, 1:]
        
        # Look Ahead Mask
        seq_len = tgt_in.size(1)
        mask = create_look_ahead_mask(seq_len, DEVICE) # [L, L]
        # Expand for batch: [B, 1, L, L]
        mask = mask.unsqueeze(0).unsqueeze(0).expand(labels.size(0), 1, seq_len, seq_len)
        
        optimizer.zero_grad()
        with autocast():
            # Pass dummy enc_output=None (handled in model)
            logits_steps = model(tgt_in, look_ahead_mask=mask) # [T, B, L, V]
            
            # Loss on average or each step
            # Here we sum loss over steps or usually just take mean logits for SNN?
            # Let's sum loss for stronger signal
            loss = 0
            for t in range(logits_steps.shape[0]):
                loss += criterion(logits_steps[t].reshape(-1, VOCAB_SIZE), tgt_out.reshape(-1))
            loss = loss / logits_steps.shape[0]
            
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()
        
        steps += 1
        pbar.update(1)
        pbar.set_postfix(loss=f"{loss.item():.4f}")
        
        if steps % 5000 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), "checkpoints/decoder_pretrain_best.pt")
            
    print("Decoder Pretraining Completed.")

if __name__ == "__main__":
    main()
