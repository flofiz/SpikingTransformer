import os
import time
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from torch.cuda.amp import autocast, GradScaler
from models_pretraining import PretrainEncoderModel

# CONFIG
DEVICE = "cuda:0"
BATCH_SIZE = 128
NUM_EPOCHS = 10
LR = 1e-3
IMG_SIZE = 224
NUM_WORKERS = 8
GRAD_CLIP = 1.0

def main():
    print(f"Starting Encoder Pretraining on {DEVICE}")
    
    # Transform: ImageNet Standard
    transform = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Dataset - Try local ImageNet first, otherwise fake for dry-run
    try:
        train_ds = datasets.ImageFolder(root="/path/to/imagenet/train", transform=transform)
        print("Loaded ImageNet from disk.")
    except:
        print("⚠️ ImageNet not found at /path/to/imagenet/train. Using FakeData for verification.")
        train_ds = datasets.FakeData(size=1000, image_size=(3, IMG_SIZE, IMG_SIZE), num_classes=1000, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
    
    model = PretrainEncoderModel(num_classes=1000).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        acc_sum = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"[Enc GPU:0] Epoch {epoch+1}/{NUM_EPOCHS}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            with autocast():
                logits_steps = model(imgs) # [T, B, 1000]
                mean_logits = logits_steps.mean(dim=0) # [B, 1000]
                loss = criterion(mean_logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Metrics
            acc = (mean_logits.argmax(dim=-1) == labels).float().mean().item()
            running_loss += loss.item()
            acc_sum += acc
            total += 1
            
            pbar.set_postfix(loss=f"{running_loss/total:.4f}", acc=f"{acc_sum/total:.2%}", lr=f"{scheduler.get_last_lr()[0]:.2e}")
        
        # Save Checkpoint
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), "checkpoints/encoder_pretrain_best.pt")
        print(f"Saved checkpoint to checkpoints/encoder_pretrain_best.pt")

if __name__ == "__main__":
    main()
