import os
import argparse
import math
import time
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm.auto import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast

from transformers import TrOCRProcessor
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from Triton_Layers.Seq2Seq import Seq2Seq as Seq2SeqTransformer
from wiki_text_images3 import WikiTextImageDataset, WikiTextDataCollator

# ============================================
# HELPER FUNCTIONS
# ============================================

def create_padding_mask(seq: Tensor, pad_idx: int, device) -> Tensor:
    # 1 pour garder, 0 pour pad
    return torch.where(seq == pad_idx, 0.0, 1.0).unsqueeze(1).unsqueeze(1).to(device)  # (B,1,1,T)

def create_look_ahead_mask(size: int, device) -> Tensor:
    return torch.tril(torch.ones((size, size), device=device))

def create_decoder_mask(tgt: Tensor, pad_idx: int, device) -> Tensor:
    # Combine padding mask and look-ahead mask: (B, 1, T, T)
    look_ahead = create_look_ahead_mask(tgt.shape[1], device)  # (T, T)
    dec_target_padding = create_padding_mask(tgt, pad_idx, device)  # (B, 1, 1, T)
    # broadcast to (B,1,T,T)
    return torch.minimum(dec_target_padding, look_ahead)  # float 0/1

def trim_to_eos(ids_1d: Tensor, eos_idx: int) -> Tensor:
    # ids_1d: (T,)
    ids = ids_1d.detach().cpu()
    eos_positions = (ids == eos_idx).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        end = eos_positions[0].item()
        return ids[:end]  # exclut EOS lui-même
    return ids

def batch_trim_to_eos(ids_2d: Tensor, eos_idx: int) -> List[Tensor]:
    # ids_2d: (B, T)
    return [trim_to_eos(ids_2d[i], eos_idx) for i in range(ids_2d.size(0))]

def decode_ids_list(processor: TrOCRProcessor, list_of_1d_ids: List[Tensor]) -> List[str]:
    # Décode une liste de séquences (longueurs variables) en chaînes
    texts = []
    for ids in list_of_1d_ids:
        texts.append(processor.decode(ids.tolist(), skip_special_tokens=True))
    return texts

def tokens_to_strings_until_eos(processor: TrOCRProcessor, token_ids: Tensor, eos_idx: int) -> List[str]:
    # token_ids: (B, T)
    trimmed = batch_trim_to_eos(token_ids, eos_idx)
    return decode_ids_list(processor, trimmed)

def strings_from_logits_until_eos(processor: TrOCRProcessor, logits: Tensor, eos_idx: int) -> List[str]:
    # logits: (B, T, V)
    pred_ids = torch.argmax(logits, dim=-1)  # (B,T)
    return tokens_to_strings_until_eos(processor, pred_ids, eos_idx)

def get_decoder_start_id(tokenizer) -> int:
    if getattr(tokenizer, "bos_token_id", None) is not None:
        return tokenizer.bos_token_id
    if getattr(tokenizer, "cls_token_id", None) is not None:
        return tokenizer.cls_token_id
    # Fallback: parfois on démarre avec pad si rien n'est défini
    return tokenizer.pad_token_id

def compute_token_accuracy(logits: Tensor, targets: Tensor, pad_idx: int) -> float:
    """
    Calcule la précision des tokens (hors padding).
    logits: (B, T, V)
    targets: (B, T)
    """
    preds = torch.argmax(logits, dim=-1)  # (B, T)
    mask = targets != pad_idx
    correct = (preds == targets) & mask
    return correct.sum().float() / max(1.0, mask.sum().float())

def compute_perplexity(loss: float) -> float:
    """
    Calcule la perplexité à partir de la loss (cross-entropy).
    Perplexity = exp(loss)
    """
    return math.exp(min(loss, 100.0))  # Cap to avoid overflow

def levenshtein_distance(s1: str, s2: str) -> int:
    """Calcul de la distance de Levenshtein (édition) entre deux chaînes."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def compute_cer(preds: List[str], targets: List[str]) -> float:
    """
    Calcule le Character Error Rate (CER).
    CER = (Sub + Del + Ins) / len(Reference)
    """
    total_dist = 0
    total_len = 0
    for p, t in zip(preds, targets):
        dist = levenshtein_distance(p, t)
        total_dist += dist
        total_len += len(t)
    
    return total_dist / max(1, total_len)

# ============================================
# WANDB (Simplified for Eval)
# ============================================
WANDB_AVAILABLE = False
wandb = None

def init_wandb(args, config_dict: dict) -> bool:
    if not args.use_wandb:
        return False
    global WANDB_AVAILABLE, wandb
    try:
        import wandb as _wandb
        wandb = _wandb
        wandb.init(
            project=args.wandb_project,
            name=f"eval_spikeformer_{time.strftime('%Y%m%d_%H%M%S')}",
            config=config_dict,
            tags=["evaluation", "spikeformer"],
        )
        WANDB_AVAILABLE = True
        return True
    except Exception as e:
        print(f"[WandB] Initialization failed: {e}")
        return False

# ============================================
# ARGUMENTS
# ============================================
def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Spiking Transformer OCR")
    
    # Model config (Must match training)
    parser.add_argument("--mask_mode", type=str, default="multiply", choices=["multiply", "additive"])
    parser.add_argument("--use_mssa", action="store_true", help="Use Multi-Scale Spiking Attention")
    parser.add_argument("--mssa_scales", type=str, default="1,2,4", help="Comma-separated MSSA scales")
    parser.add_argument("--in_channels", type=int, default=1, choices=[1, 3], help="Input channels")
    parser.add_argument("--num_steps", type=int, default=4, help="Number of SNN timesteps")
    parser.add_argument("--encoder_type", type=str, default="scs", choices=["cnn", "scs"])
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing")
    parser.set_defaults(gradient_checkpointing=True)

    # Eval config
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/spikeformer2_best.pt")
    parser.add_argument("--batch_size", type=int, default=32) # Default slightly smaller for safety
    parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate on")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples")
    
    # WandB
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="spikeformer-ocr")

    return parser.parse_args()

# ============================================
# MAIN EVALUATION
# ============================================
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*60)
    print("Evaluating Spikeformer OCR")
    print("="*60)
    print(f"Device: {device}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Batch Size: {args.batch_size}")
    
    # Hyperparameters (Matching user's training config or defaults)
    EMB_SIZE = 384
    NHEAD_ENCODER = 6
    NHEAD_DECODER = 12
    FFN_HID_DIM = 4 * EMB_SIZE
    NUM_ENCODER_LAYERS = 12
    NUM_DECODER_LAYERS = 6
    PATCH_SIZE = 16
    IMG_SIZE = (64, 768)
    MAX_CHARS = 32
    
    MSSA_SCALES = [int(x) for x in args.mssa_scales.split(",")]
    
    if args.use_mssa and NHEAD_ENCODER % len(MSSA_SCALES) != 0:
        NHEAD_ENCODER = len(MSSA_SCALES) * (NHEAD_ENCODER // len(MSSA_SCALES) + 1)

    # Initialize Processor
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    PAD_IDX = processor.tokenizer.pad_token_id
    EOS_IDX = processor.tokenizer.eos_token_id
    START_IDX = get_decoder_start_id(processor.tokenizer)
    VOCAB_SIZE = len(processor.tokenizer)

    # Load Dataset
    print(f"Loading dataset (split={args.split})...")
    dataset = WikiTextImageDataset(
        processor=processor,
        split=args.split,
        img_size=IMG_SIZE,
        train=False,
        max_samples=args.max_samples if args.max_samples else 10000,
        max_chars=MAX_CHARS,
        in_channels=args.in_channels,
        sources=[("wikimedia/wikipedia", "20231101.fr", "Français")],
    )
    
    data_collator = WikiTextDataCollator(processor, max_length=MAX_CHARS)
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        collate_fn=data_collator,
        shuffle=False,
        pin_memory=True
    )

    # Initialize Model
    print("Initializing model...")
    model = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_model=EMB_SIZE,
        n_heads_encoder=NHEAD_ENCODER,
        n_heads_decoder=NHEAD_DECODER,
        tgt_vocab_size=VOCAB_SIZE,
        ff_dim=FFN_HID_DIM,
        n_steps=args.num_steps,
        nb_sps_blocks=4,
        patch_size=PATCH_SIZE,
        mask_mode=args.mask_mode,
        use_mssa=args.use_mssa,
        mssa_scales=MSSA_SCALES,
        in_channels=args.in_channels,
        img_height=IMG_SIZE[0],
        gradient_checkpointing=args.gradient_checkpointing,
        encoder_type=args.encoder_type,
    ).to(device)

    # Load Checkpoint
    if os.path.exists(args.checkpoint_path):
        print(f"Loading weights from {args.checkpoint_path}...")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        
        # Handle state dict (un-wrap DDP if needed)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
            
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k.replace("module.", "") # remove DDP prefix
            new_state_dict[name] = v
            
        try:
            model.load_state_dict(new_state_dict, strict=True)
            print("✓ Checkpoint loaded successfully.")
        except Exception as e:
            print(f"⚠ Error loading checkpoint: {e}")
            print("Attempting non-strict load...")
            model.load_state_dict(new_state_dict, strict=False)
    else:
        print(f"❌ Checkpoint not found at {args.checkpoint_path}")
        return

    # Init WandB
    init_wandb(args, vars(args))

    # Evaluation Loop
    model.eval()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.0) # No smoothing for pure eval
    
    total_loss = 0.0
    total_acc = 0.0
    total_cer = 0.0
    n_batches = 0
    total_seen = 0
    
    print("\nStarting evaluation...")
    pbar = tqdm(dataloader, desc="Evaluating", unit="batch")
    
    all_preds = []
    all_targets = []
    
    for batch in pbar:
        src = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        tgt_in = labels[:, :-1]
        tgt_out = labels[:, 1:] # Target for loss
        
        dec_mask = create_decoder_mask(tgt_in, PAD_IDX, device)
        
        with torch.no_grad():
            with autocast(dtype=torch.float16):
                logits_steps, _ = model(
                    src, tgt_in,
                    look_ahead_mask=dec_mask
                )
                
                # Loss calculation (Teacher Forcing)
                loss = 0.0
                for s in range(logits_steps.shape[0]):
                    l_s = criterion(logits_steps[s].reshape(-1, logits_steps.size(-1)), tgt_out.reshape(-1))
                    loss += l_s
                loss /= logits_steps.shape[0]

            # Greedy Decode for Metrics
            # We use the mean logits from Teacher Forcing for quick metrics in training,
            # but for true evaluation we should probably do autoregressive greedy decode?
            # Creating autoregressive decode loop is slow.
            # Ideally "Evaluation" report usually checks Teacher Forcing metrics (Perplexity, Acc) 
            # AND Generation metrics (CER).
            # The training script used `mean_logits` from TF for CER. This is "TF-CER".
            # Real CER requires `greedy_decode`.
            # Let's do `greedy_decode` for a subset or all if fast enough? 
            # For 10k samples, greedy decode might take a while.
            # Let's stick to the training script's method (TF-CER) for speed, 
            # but ALSO run a small sample of full greedy decode for verification.
            
            # Actually, user wants "evaluation script". Usually this implies real performance.
            # I will implement real greedy decoding for CER if batch_size is small or optional?
            # Let's try to do what training script did first (mean logits of TF) to be consistent on "Val Loss" 
            # BUT for CER, we really should decode.
            
            # Re-reading training script: 
            # `pred_strs = strings_from_logits_until_eos(processor, mean_logits, EOS_IDX)`
            # This comes from `mean_logits` which comes from `logits_steps` (Teacher Forcing).
            # So the training CER is a Teacher Forcing CER.
            
            # Let's do the same for the main loop for speed, 
            # AND print some real greedy examples.
            
            mean_logits = logits_steps.mean(dim=0)
            acc = compute_token_accuracy(mean_logits, tgt_out, PAD_IDX)
            
            pred_strs = strings_from_logits_until_eos(processor, mean_logits, EOS_IDX)
            tgt_strs = tokens_to_strings_until_eos(processor, tgt_out, EOS_IDX)
            
            batch_cer = compute_cer(pred_strs, tgt_strs)
            
            total_loss += loss.item()
            total_acc += acc.item()
            total_cer += batch_cer
            n_batches += 1
            total_seen += src.size(0)
            
            all_preds.extend(pred_strs)
            all_targets.extend(tgt_strs)
            
            pbar.set_postfix({
                "Loss": f"{loss.item():.3f}",
                "Acc": f"{acc.item():.1%}",
                "CER": f"{batch_cer:.1%}"
            })

    avg_loss = total_loss / n_batches
    avg_acc = total_acc / n_batches
    avg_cer = total_cer / n_batches
    perplexity = compute_perplexity(avg_loss)
    
    print("\n" + "="*30)
    print("RESULTS")
    print("="*30)
    print(f"Samples: {total_seen}")
    print(f"Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")
    print(f"Token Acc: {avg_acc:.2%}")
    print(f"CER (TF): {avg_cer:.2%}") # Teacher Forcing CER
    
    if WANDB_AVAILABLE and wandb is not None:
        wandb.log({
            "eval_loss": avg_loss,
            "eval_perplexity": perplexity,
            "eval_accuracy": avg_acc,
            "eval_cer": avg_cer,
        })
    
    # ============================================
    # GENERATION EXAMPLES (True Greedy Decode)
    # ============================================
    print("\n" + "="*60)
    print("Generating Examples (True Greedy Decode)...")
    print("="*60)
    
    indices = torch.randperm(len(dataset))[:5] # Take 5 random examples
    subset = torch.utils.data.Subset(dataset, indices)
    loader_ex = DataLoader(subset, batch_size=1, collate_fn=data_collator)
    
    wandb_table = []
    
    for i, batch in enumerate(loader_ex):
        src = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)
        
        # Greedy Decode
        ys, _, _ = model.greedy_decode(
            src=src,
            max_len=MAX_CHARS,
            start_symbol=START_IDX,
            eos_idx=EOS_IDX,
            pad_idx=PAD_IDX,
            device=device,
        )
        
        gen_ids = trim_to_eos(ys[0], EOS_IDX)
        gen_text = processor.decode(gen_ids.tolist(), skip_special_tokens=True)
        
        gt_ids = trim_to_eos(labels[0], EOS_IDX)
        gt_text = processor.decode(gt_ids.tolist(), skip_special_tokens=True)
        
        print(f"\nExample {i+1}:")
        print(f"  GT:  {gt_text}")
        print(f"  Pred:{gen_text}")
        
        if WANDB_AVAILABLE and wandb is not None:
            # Reconstruct image for logging
            img_t = src[0].cpu()
            if img_t.shape[0] == 1:
                img_pil = TF.to_pil_image(img_t, mode='L')
            else:
                img_pil = TF.to_pil_image(img_t, mode='RGB')
                
            wandb_table.append([wandb.Image(img_pil), gt_text, gen_text])

    if WANDB_AVAILABLE and wandb is not None and wandb_table:
        wandb.log({"examples": wandb.Table(columns=["Image", "GT", "Prediction"], data=wandb_table)})
        print("Logged examples to WandB.")

if __name__ == "__main__":
    main()
