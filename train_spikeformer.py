import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import argparse
import math
import time
from pathlib import Path
from typing import Tuple, List, Literal, Optional
from tqdm.auto import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import TrOCRProcessor
import torch.nn.functional as F
from Triton_Layers.Seq2Seq import Seq2Seq as Seq2SeqTransformer
from wiki_text_images3 import WikiTextImageDataset, WikiTextDataCollator


# ============================================
# WANDB INTEGRATION
# ============================================
WANDB_AVAILABLE = False
wandb = None

def init_wandb(args, config_dict: dict) -> bool:
    """
    Initialize WandB from API key file.
    
    Looks for API key in:
    1. Environment variable WANDB_API_KEY
    2. File: wandb_key.txt in current directory
    3. File: ~/.wandb_key
    
    Returns True if WandB was initialized successfully.
    """
    global WANDB_AVAILABLE, wandb
    
    if not args.use_wandb:
        return False
    
    try:
        import wandb as _wandb
        wandb = _wandb
    except ImportError:
        print("[WandB] wandb package not installed. Run: pip install wandb")
        return False
    
    # Find API key
    api_key = os.environ.get("WANDB_API_KEY")
    
    if not api_key:
        key_files = [
            Path("wandb_key.txt"),
            Path.home() / ".wandb_key",
            Path.home() / "wandb_key.txt",
        ]
        
        for key_file in key_files:
            if key_file.exists():
                try:
                    api_key = key_file.read_text().strip()
                    print(f"[WandB] Loaded API key from {key_file}")
                    break
                except Exception as e:
                    print(f"[WandB] Failed to read {key_file}: {e}")
    
    if not api_key:
        print("[WandB] No API key found. Create wandb_key.txt with your API key.")
        print("        Or set WANDB_API_KEY environment variable.")
        return False
    
    try:
        wandb.login(key=api_key, relogin=True)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"spikeformer_{time.strftime('%Y%m%d_%H%M%S')}",
            config=config_dict,
            tags=["spiking-transformer", "ocr"],
        )
        WANDB_AVAILABLE = True
        print(f"[WandB] Initialized: {wandb.run.url}")
        return True
    except Exception as e:
        print(f"[WandB] Initialization failed: {e}")
        return False


def log_metrics(metrics: dict, step: int):
    """Log metrics to WandB if available."""
    if WANDB_AVAILABLE and wandb is not None:
        wandb.log(metrics, step=step)


def finish_wandb():
    """Finish WandB run."""
    if WANDB_AVAILABLE and wandb is not None:
        wandb.finish()


# ============================================
# TRAINING CONFIGURATION
# ============================================
def get_training_config(args):
    """
    Returns training configuration based on GPU and arguments.
    Auto-detects GPU type if batch_size not specified.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        # Auto-detect based on GPU VRAM
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            if total_vram >= 70:  # A100 80GB
                batch_size = 96
            elif total_vram >= 30:  # A100 40GB
                batch_size = 64
            elif total_vram >= 20:  # RTX 3090/4090
                batch_size = 48
            else:  # RTX 4070 Ti Super 16GB or smaller
                batch_size = 24
            print(f"[AutoConfig] Detected {total_vram:.1f}GB VRAM -> batch_size={batch_size}")
        else:
            batch_size = 8
    
    num_workers = 16 if batch_size >= 64 else 8
    
    return {
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }


def get_curriculum_config(step: int, total_steps: int) -> dict:
    """
    Returns curriculum configuration based on training progress.
    Starts with shorter sequences and smaller batch sizes, gradually increases.
    """
    progress = step / max(1, total_steps)
    
    if progress < 0.15:      # Phase 1: 0-15%
        return {"max_chars": 32, "batch_multiplier": 2.0}
    elif progress < 0.35:    # Phase 2: 15-35%
        return {"max_chars": 48, "batch_multiplier": 1.5}
    elif progress < 0.60:    # Phase 3: 35-60%
        return {"max_chars": 80, "batch_multiplier": 1.0}
    else:                    # Phase 4: 60-100%
        return {"max_chars": 128, "batch_multiplier": 0.75}


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


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


def compute_gradient_norm(model: nn.Module) -> float:
    """Compute the total gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Train Spiking Transformer OCR")
    
    # Training config
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (auto-detected if not specified)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    
    # Model config
    parser.add_argument("--mask_mode", type=str, default="multiply", choices=["multiply", "additive"],
                        help="Masking mode for attention: 'multiply' or 'additive'")
    parser.add_argument("--use_mssa", action="store_true", help="Use Multi-Scale Spiking Attention")
    parser.add_argument("--mssa_scales", type=str, default="1,2,4", help="Comma-separated MSSA scales")
    parser.add_argument("--in_channels", type=int, default=3, choices=[1, 3], 
                        help="Input channels: 1 for grayscale, 3 for RGB")
    parser.add_argument("--num_steps", type=int, default=8, help="Number of SNN timesteps")

    # Optimization
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing",
                        help="Disable gradient checkpointing (enabled by default)")
    parser.add_argument("--no_gradient_checkpointing", action="store_false", dest="gradient_checkpointing",
                        help="Disable gradient checkpointing (enabled by default)")
    parser.set_defaults(gradient_checkpointing=True)

    parser.add_argument("--compile", action="store_true", help="Enable torch.compile() for faster training")
    
    # Image config - State-of-the-art sizes for OCR
    parser.add_argument("--img_height", type=int, default=64, help="Image height (SotA: 64 for text lines)")
    parser.add_argument("--img_width", type=int, default=768, help="Image width (SotA: 768 for text lines)")
    
    # Curriculum learning
    parser.add_argument("--use_curriculum", action="store_true", help="Enable curriculum learning")
    
    # WandB logging
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="spikeformer-ocr", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name (auto-generated if not set)")
    
    return parser.parse_args()


def train():
    args = parse_args()
    config = get_training_config(args)
    
    # ============================================
    # HYPERPARAMETERS - Optimized
    # ============================================
    EMB_SIZE = 384
    NHEAD = 6  # Must be divisible by len(mssa_scales) if use_mssa=True
    FFN_HID_DIM = 4 * EMB_SIZE
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    NUM_STEPS = args.num_steps
    LR = args.lr  # Reduced from 1e-3 to 5e-4 for SNN stability
    BATCH_SIZE = config["batch_size"]
    NUM_EPOCHS = args.epochs
    IMG_SIZE = (args.img_height, args.img_width)  # State-of-the-art: (64, 768) for text lines
    LOG_EVERY = 100
    EVAL_EVERY = 2000
    LOG_PRINT_EVERY = 1000
    MAX_CHARS = 128 
    GRAD_CLIP_NORM = 1.0
    WEIGHT_DECAY = 0.01
    MASK_MODE = args.mask_mode
    USE_MSSA = args.use_mssa
    MSSA_SCALES = [int(x) for x in args.mssa_scales.split(",")]
    IN_CHANNELS = args.in_channels
    IN_CHANNELS = args.in_channels
    USE_CURRICULUM = args.use_curriculum
    GRADIENT_CHECKPOINTING = args.gradient_checkpointing

    # Adjust NHEAD for MSSA compatibility
    if USE_MSSA and NHEAD % len(MSSA_SCALES) != 0:
        NHEAD = len(MSSA_SCALES) * (NHEAD // len(MSSA_SCALES) + 1)
        print(f"[MSSA] Adjusted NHEAD to {NHEAD} for compatibility with {len(MSSA_SCALES)} scales")

    # Create config dict for WandB
    wandb_config = {
        "emb_size": EMB_SIZE,
        "n_heads": NHEAD,
        "ffn_dim": FFN_HID_DIM,
        "num_encoder_layers": NUM_ENCODER_LAYERS,
        "num_decoder_layers": NUM_DECODER_LAYERS,
        "num_steps": NUM_STEPS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "max_chars": MAX_CHARS,
        "grad_clip_norm": GRAD_CLIP_NORM,
        "weight_decay": WEIGHT_DECAY,
        "mask_mode": MASK_MODE,
        "use_mssa": USE_MSSA,
        "mssa_scales": MSSA_SCALES,
        "in_channels": IN_CHANNELS,
        "in_channels": IN_CHANNELS,
        "use_curriculum": USE_CURRICULUM,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
    }
    
    # Initialize WandB
    init_wandb(args, wandb_config)

    print("="*60)
    print("Configuration:")
    print("="*60)
    print(f"  Device: {config['device']}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LR}")
    print(f"  Image size: {IMG_SIZE}")
    print(f"  Input channels: {IN_CHANNELS} ({'RGB' if IN_CHANNELS == 3 else 'Grayscale'})")
    print(f"  Mask mode: {MASK_MODE}")
    print(f"  Use MSSA: {USE_MSSA}")
    if USE_MSSA:
        print(f"  MSSA scales: {MSSA_SCALES}")
    print(f"  NUM_STEPS: {NUM_STEPS}")
    print(f"  NUM_STEPS: {NUM_STEPS}")
    print(f"  Curriculum learning: {USE_CURRICULUM}")
    print(f"  Gradient Checkpointing: {GRADIENT_CHECKPOINTING}")
    print(f"  WandB logging: {WANDB_AVAILABLE}")
    print("="*60 + "\n")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    PAD_IDX = processor.tokenizer.pad_token_id
    EOS_IDX = processor.tokenizer.eos_token_id
    START_IDX = get_decoder_start_id(processor.tokenizer)
    VOCAB_SIZE = len(processor.tokenizer)

    # ============================================
    # DATASETS
    # ============================================
    train_ds = WikiTextImageDataset(
        processor=processor,
        split="train",
        img_size=IMG_SIZE,
        train=True,
        max_samples=50_000_000,
        max_chars=MAX_CHARS,
        cache_size=100,
        article_rotation_interval=500_000,
        in_channels=IN_CHANNELS,  # Added: RGB or grayscale
        sources=[
            ("wikimedia/wikipedia", "20231101.fr", "Français"),
        ],
    )

    val_ds = WikiTextImageDataset(
        processor=processor,
        split="test",
        img_size=IMG_SIZE,
        train=False,
        max_samples=10_000,
        max_chars=MAX_CHARS,
        in_channels=IN_CHANNELS,  # Added: RGB or grayscale
        sources=[
            ("wikimedia/wikipedia", "20231101.fr", "Français"),
        ]
    )

    # Data Collator for dynamic padding
    data_collator = WikiTextDataCollator(processor, max_length=MAX_CHARS)

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=config["num_workers"], 
        prefetch_factor=8,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collator
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=config["num_workers"], 
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collator
    )

    # ============================================
    # MODEL - with new parameters
    # ============================================
    model = Seq2SeqTransformer(
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        d_model=EMB_SIZE,
        n_heads=NHEAD,
        tgt_vocab_size=VOCAB_SIZE,
        ff_dim=FFN_HID_DIM,
        n_steps=NUM_STEPS,
        nb_sps_blocks=4,
        patch_size=4,
        mask_mode=MASK_MODE,       # New: configurable mask mode
        use_mssa=USE_MSSA,         # New: Multi-Scale Spiking Attention
        mssa_scales=MSSA_SCALES,   # New: MSSA scales
        in_channels=IN_CHANNELS,   # New: RGB or grayscale input
        img_height=IMG_SIZE[0],    # New: image height for channel computation
        gradient_checkpointing=GRADIENT_CHECKPOINTING, # New: gradient checkpointing
    ).to(config["device"])

    # Compile model if requested
    if args.compile:
        print("[Torch.Compile] Compiling model... (backend='inductor')")
        model = torch.compile(model, backend="inductor")

    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    # ============================================
    # LOSS, OPTIMIZER, SCHEDULER
    # ============================================
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR, 
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=WEIGHT_DECAY
    )
    
    scaler = GradScaler()
    total_steps = NUM_EPOCHS * len(train_loader)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4
    )

    # ============================================
    # LOSS FUNCTION
    # ============================================
    def step_loss(logits_steps: Tensor, tgt_out: Tensor) -> Tensor:
        loss = 0.0
        for s in range(logits_steps.shape[0]):
            logits = logits_steps[s]
            loss = loss + criterion(
                logits.reshape(-1, logits.size(-1)), 
                tgt_out.reshape(-1)
            )
        return loss / logits_steps.shape[0]

    # ============================================
    # EVALUATION FUNCTION
    # ============================================
    @torch.no_grad()
    def evaluate() -> Tuple[float, float, float, float, float]:
        model.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        eval_cer = 0.0
        n_batches = 0
        
        eval_pbar = tqdm(val_loader, desc="  Evaluation  ", unit="batch")
        for batch in eval_pbar:
            src = batch["pixel_values"].to(config["device"], non_blocking=True)
            labels = batch["labels"].to(config["device"], non_blocking=True)
            tgt_in = labels[:, :-1]
            tgt_out = labels[:, 1:]
            dec_mask = create_decoder_mask(tgt_in, PAD_IDX, config["device"])
            
            with autocast(dtype=torch.float16):
                logits_steps, _ = model(
                    src, tgt_in, 
                    enc_padding_mask=None, 
                    look_ahead_mask=dec_mask, 
                    dec_padding_mask=None
                )
                loss = step_loss(logits_steps, tgt_out)
            
            mean_logits = logits_steps.mean(dim=0)
            acc = compute_token_accuracy(mean_logits, tgt_out, PAD_IDX)
            pred_strs = strings_from_logits_until_eos(processor, mean_logits, EOS_IDX)
            tgt_strs = tokens_to_strings_until_eos(processor, tgt_out, EOS_IDX)
            cer = compute_cer(pred_strs, tgt_strs)
            
            eval_loss += loss.item()
            eval_acc += acc.item()
            eval_cer += cer
            n_batches += 1
            
            eval_pbar.set_postfix({"L": f"{loss.item():.3f}", "A": f"{acc.item():.1%}", "C": f"{cer:.1%}"})
        
        avg_loss = eval_loss / max(1, n_batches)
        avg_acc = eval_acc / max(1, n_batches)
        avg_cer = eval_cer / max(1, n_batches)
        avg_ppl = compute_perplexity(avg_loss)
        
        return avg_loss, avg_acc, avg_cer, avg_ppl, scheduler.get_last_lr()[0]

    # ============================================
    # PRINT EXAMPLES FUNCTION
    # ============================================
    @torch.no_grad()
    def print_examples(batch_src, batch_labels, logits_steps, max_examples=3):
        logits = logits_steps.mean(dim=0)
        pred_tf_str = strings_from_logits_until_eos(processor, logits, EOS_IDX)
        gt_str = tokens_to_strings_until_eos(processor, batch_labels, EOS_IDX)

        print("\n" + "="*60)
        print("Exemples (Teacher Forcing vs Greedy Decoding):")
        print("="*60)
        nb = min(max_examples, batch_src.size(0))
        for i in range(nb):
            ys, _, _ = model.greedy_decode(
                src=batch_src[i],
                max_len=batch_labels.size(1),
                start_symbol=START_IDX,
                eos_idx=EOS_IDX,
                pad_idx=PAD_IDX,
                device=config["device"],
            )
            gen_trim = trim_to_eos(ys[0], EOS_IDX)
            gen_str = processor.decode(gen_trim.tolist(), skip_special_tokens=True)

            print(f"\n[Example {i+1}]")
            print(f"  GT (Ground Truth): {gt_str[i]}")
            print(f"  TF (Teacher Force): {pred_tf_str[i]}")
            print(f"  GD (Greedy Decode): {gen_str}")
        print("="*60 + "\n", flush=True)

    # ============================================
    # SANITY CHECK
    # ============================================
    print("="*60)
    print("Sanity Check: Test de génération avant entraînement...")
    print("="*60)
    try:
        model.eval()
        with torch.no_grad():
            first_batch = next(iter(train_loader))
            src0 = first_batch["pixel_values"].to(config["device"], non_blocking=True)[0]
            labels0 = first_batch["labels"].to(config["device"], non_blocking=True)[0]
            _ys, _attn, _out = model.greedy_decode(
                src=src0,
                max_len=labels0.size(0),
                start_symbol=START_IDX,
                eos_idx=EOS_IDX,
                pad_idx=PAD_IDX,
                device=config["device"],
            )
            gen_str = processor.decode(trim_to_eos(_ys[0], EOS_IDX).tolist(), skip_special_tokens=True)
            gt_str = processor.decode(trim_to_eos(labels0, EOS_IDX).tolist(), skip_special_tokens=True)
            print(f"✅ Sanity check PASSED")
            print(f"   Ground Truth: {gt_str}")
            print(f"   Generated (random init): {gen_str}", flush=True)
    except Exception as e:
        print(f"❌ Sanity check FAILED: {e}")
        raise
    print("="*60 + "\n")

    # ============================================
    # TRAINING LOOP
    # ============================================
    global_step = 0
    best_val = float("inf")

    print("="*60)
    print("Début de l'entraînement")
    print("="*60)
    print(f"Total steps: {total_steps}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Learning rate: {LR} (warmup 10%, cosine decay)")
    print(f"Gradient clipping: {GRAD_CLIP_NORM}")
    print(f"Weight decay: {WEIGHT_DECAY}")
    print(f"Label smoothing: 0.1")
    print("="*60 + "\n", flush=True)

    # Check all params are in optimizer
    model_params = set(model.parameters())
    optimizer_params = set()
    for group in optimizer.param_groups:
        optimizer_params.update(group['params'])

    missing = model_params - optimizer_params
    if missing:
        print(f"❌ {len(missing)} paramètres ne sont PAS dans l'optimizer!")
    else:
        print("✅ Tous les paramètres sont dans l'optimizer")

    # ============================================
    # RESUME FROM CHECKPOINT
    # ============================================
    ckpt_path = "checkpoints/spikeformer2_best.pt"
    start_epoch = 1
    
    if os.path.exists(ckpt_path):
        print(f"\n{'='*60}")
        print(f"🔄 Checkpoint trouvé: {ckpt_path}")
        print("Chargement en cours...")
        
        checkpoint = torch.load(ckpt_path, map_location=config["device"])
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        print(f"Model loaded: {msg}")
        
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Optimizer loaded")
            
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
            print("Scheduler loaded")
            
        global_step = checkpoint.get("step", 0)
        best_val = checkpoint.get("val_loss", float("inf"))
        
        steps_per_epoch = len(train_loader)
        if steps_per_epoch > 0:
            start_epoch = (global_step // steps_per_epoch) + 1
            
        print(f"Resuming from step {global_step} (Epoch {start_epoch}), Best Val Loss: {best_val:.4f}")
        print(f"{'='*60}\n", flush=True)
    else:
        print(f"\n⚠️ Aucun checkpoint trouvé à {ckpt_path}. Démarrage de zéro.\n", flush=True)

    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        t0 = time.time()
        pbar = tqdm(
            train_loader, 
            total=len(train_loader), 
            desc=f"Epoch {epoch}/{NUM_EPOCHS}", 
            unit="batch"
        )
        
        for batch in pbar:
            global_step += 1
            
            # Curriculum learning
            if USE_CURRICULUM:
                curriculum = get_curriculum_config(global_step, total_steps)
                # Note: Dynamic batch size change not implemented in this loop
                # but max_chars can be used in future data loader updates
            
            optimizer.zero_grad(set_to_none=True)

            src = batch["pixel_values"].to(config["device"], non_blocking=True)
            labels = batch["labels"].to(config["device"], non_blocking=True)
            tgt_in = labels[:, :-1]
            tgt_out = labels[:, 1:]
            
            # Compute average number of chars in batch
            num_chars = (tgt_out != PAD_IDX).sum().item()
            avg_chars_per_sample = num_chars / tgt_out.size(0)

            dec_mask = create_decoder_mask(tgt_in, PAD_IDX, config["device"])

            with autocast(dtype=torch.float16):
                logits_steps, _ = model(
                    src, tgt_in, 
                    enc_padding_mask=None, 
                    look_ahead_mask=dec_mask, 
                    dec_padding_mask=None
                )
                loss = step_loss(logits_steps, tgt_out)
                mean_logits = logits_steps.mean(dim=0)
                acc = compute_token_accuracy(mean_logits, tgt_out, PAD_IDX)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Compute gradient norm before clipping
            grad_norm = compute_gradient_norm(model)
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            current_lr = scheduler.get_last_lr()[0]
            current_loss = loss.item()
            current_acc = acc.item()
            current_ppl = compute_perplexity(current_loss)
        
            running_loss += current_loss
            running_acc += current_acc

            # Log to WandB every step (or less frequently for efficiency)
            if global_step % 10 == 0:
                log_metrics({
                    "train/loss": current_loss,
                    "train/accuracy": current_acc,
                    "train/perplexity": current_ppl,
                    "train/grad_norm": grad_norm,
                    "train/learning_rate": current_lr,
                    "train/batch_size": BATCH_SIZE,
                    "train/avg_chars": avg_chars_per_sample,
                    "train/epoch": epoch,
                }, step=global_step)

            if global_step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                avg_acc = running_acc / LOG_EVERY
                avg_ppl = compute_perplexity(avg_loss)
                running_loss = 0.0
                running_acc = 0.0
                pbar.set_postfix({
                    "Loss": f"{avg_loss:.4f}", 
                    "Acc": f"{avg_acc:.2%}",
                    "PPL": f"{avg_ppl:.1f}",
                    "LR": f"{current_lr:.2e}"
                })
            
            if global_step % LOG_PRINT_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    print_examples(src, labels, logits_steps, max_examples=3)
                model.train()

            if global_step % EVAL_EVERY == 0:
                val_loss, val_acc, val_cer, val_ppl, cur_lr = evaluate()
                print(f"\n{'='*60}")
                print(f"📊 Evaluation @ step {global_step}/{total_steps}")
                print(f"   Val Loss: {val_loss:.4f}")
                print(f"   Val Acc : {val_acc:.2%}")
                print(f"   Val CER : {val_cer:.2%}")
                print(f"   Val PPL : {val_ppl:.1f}")
                print(f"   Learning Rate: {cur_lr:.2e}")
                print(f"{'='*60}\n", flush=True)
                
                # Log validation metrics
                log_metrics({
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                    "val/cer": val_cer,
                    "val/perplexity": val_ppl,
                }, step=global_step)
                
                if val_loss < best_val:
                    best_val = val_loss
                    os.makedirs("checkpoints", exist_ok=True)
                    ckpt_path = "checkpoints/spikeformer2_best.pt"
                    torch.save({
                        "model": model.state_dict(), 
                        "optimizer": optimizer.state_dict(), 
                        "scheduler": scheduler.state_dict(),
                        "step": global_step,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_cer": val_cer,
                        "val_ppl": val_ppl,
                        "config": {
                            "emb_size": EMB_SIZE,
                            "nhead": NHEAD,
                            "num_encoder_layers": NUM_ENCODER_LAYERS,
                            "num_decoder_layers": NUM_DECODER_LAYERS,
                            "vocab_size": VOCAB_SIZE,
                            "mask_mode": MASK_MODE,
                            "use_mssa": USE_MSSA,
                            "mssa_scales": MSSA_SCALES,
                            "in_channels": IN_CHANNELS,
                            "num_steps": NUM_STEPS,
                        }
                    }, ckpt_path)
                    print(f"✅ Saved best checkpoint to {ckpt_path} (val_loss: {val_loss:.4f})\n")
                    
                    # Log best model to WandB
                    if WANDB_AVAILABLE and wandb is not None:
                        wandb.run.summary["best_val_loss"] = val_loss
                        wandb.run.summary["best_val_cer"] = val_cer
                        wandb.run.summary["best_val_ppl"] = val_ppl
                
                model.train()

        epoch_time = time.time() - t0
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} completed in {epoch_time/60:.1f} min")
        print(f"{'='*60}\n", flush=True)

    print("="*60)
    print("✅ Training completed successfully!")
    print(f"Best validation loss: {best_val:.4f}")
    print("="*60, flush=True)
    
    # Finish WandB
    finish_wandb()


if __name__ == "__main__":
    train()