import os
# Deepspeed handles visible devices, so we don't strict set it here unless needed.
# But generally we remove the hardcoded "0" to allow multi-gpu.
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
# Removed manual GradScaler/autocast for DeepSpeed
# from torch.cuda.amp import autocast, GradScaler

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)

from transformers import TrOCRProcessor
import torch.nn.functional as F
from Triton_Layers.Seq2Seq import Seq2Seq as Seq2SeqTransformer
from wiki_text_images3 import WikiTextImageDataset, WikiTextDataCollator, get_font_pool


# ============================================
# WANDB INTEGRATION
# ============================================
WANDB_AVAILABLE = False
wandb = None

def init_wandb(args, config_dict: dict) -> bool:
    """
    Initialize WandB from API key file.
    Only strictly needed on Rank 0 usually, but we check availability everywhere.
    We'll actually initialize only on rank 0 in the main loop logic for safety.
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
    
    # Only Rank 0 should initialize logging usually, checking logic in main()
    WANDB_AVAILABLE = True
    return True

def do_wandb_init_on_rank0(args, config_dict):
    """Actual WandB init called only on rank 0."""
    global wandb
    if not WANDB_AVAILABLE: 
        return
        
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
        return
    
    try:
        wandb.login(key=api_key, relogin=True)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"spikeformer_{time.strftime('%Y%m%d_%H%M%S')}",
            config=config_dict,
            tags=["spiking-transformer", "ocr", "deepspeed"],
        )
        print(f"[WandB] Initialized: {wandb.run.url}")
    except Exception as e:
        print(f"[WandB] Initialization failed: {e}")

def log_metrics(metrics: dict, step: int):
    """Log metrics to WandB if available (Safe to call, internally checks valid run)."""
    if WANDB_AVAILABLE and wandb is not None and wandb.run is not None:
        wandb.log(metrics, step=step)


def finish_wandb():
    """Finish WandB run."""
    if WANDB_AVAILABLE and wandb is not None and wandb.run is not None:
        wandb.finish()


# ============================================
# TRAINING CONFIGURATION
# ============================================
def get_training_config(args):
    """
    Returns training configuration. Use args provided values or defaults.
    With DeepSpeed, batch size is per-device typically defined in config or args.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # With DeepSpeed, global batch size is usually managed by config. 
    # But we keep local batch size logic for DataLoader.
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        # Default fallback
        batch_size = 8
        if torch.cuda.is_available():
            # Rough heuristic for A100 based on previous code, 
            # but usually with DS config "auto", we rely on args.
             if torch.cuda.get_device_properties(0).total_memory / 1e9 >= 30:
                 batch_size = 64
             else:
                 batch_size = 8
    
    num_workers = 8 if batch_size >= 64 else 4
    
    return {
        "device": device,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }

def get_curriculum_config(step: int, total_steps: int) -> dict:
    progress = step / max(1, total_steps)
    if progress < 0.15: return {"max_chars": 32, "batch_multiplier": 2.0}
    elif progress < 0.35: return {"max_chars": 48, "batch_multiplier": 1.5}
    elif progress < 0.60: return {"max_chars": 80, "batch_multiplier": 1.0}
    else: return {"max_chars": 128, "batch_multiplier": 0.75}


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True


# ============================================
# HELPER FUNCTIONS (Masks, Decoding, Metrics)
# ============================================

def create_padding_mask(seq: Tensor, pad_idx: int, device) -> Tensor:
    return torch.where(seq == pad_idx, 0.0, 1.0).unsqueeze(1).unsqueeze(1).to(device)

def create_look_ahead_mask(size: int, device) -> Tensor:
    return torch.tril(torch.ones((size, size), device=device))

def create_decoder_mask(tgt: Tensor, pad_idx: int, device) -> Tensor:
    look_ahead = create_look_ahead_mask(tgt.shape[1], device)
    dec_target_padding = create_padding_mask(tgt, pad_idx, device)
    return torch.minimum(dec_target_padding, look_ahead)

def trim_to_eos(ids_1d: Tensor, eos_idx: int) -> Tensor:
    ids = ids_1d.detach().cpu()
    eos_positions = (ids == eos_idx).nonzero(as_tuple=True)[0]
    if len(eos_positions) > 0:
        return ids[:eos_positions[0].item()]
    return ids

def batch_trim_to_eos(ids_2d: Tensor, eos_idx: int) -> List[Tensor]:
    return [trim_to_eos(ids_2d[i], eos_idx) for i in range(ids_2d.size(0))]

def decode_ids_list(processor: TrOCRProcessor, list_of_1d_ids: List[Tensor]) -> List[str]:
    texts = []
    for ids in list_of_1d_ids:
        texts.append(processor.decode(ids.tolist(), skip_special_tokens=True))
    return texts

def tokens_to_strings_until_eos(processor: TrOCRProcessor, token_ids: Tensor, eos_idx: int) -> List[str]:
    trimmed = batch_trim_to_eos(token_ids, eos_idx)
    return decode_ids_list(processor, trimmed)

def strings_from_logits_until_eos(processor: TrOCRProcessor, logits: Tensor, eos_idx: int) -> List[str]:
    pred_ids = torch.argmax(logits, dim=-1)
    return tokens_to_strings_until_eos(processor, pred_ids, eos_idx)

def get_decoder_start_id(tokenizer) -> int:
    if getattr(tokenizer, "bos_token_id", None) is not None:
        return tokenizer.bos_token_id
    if getattr(tokenizer, "cls_token_id", None) is not None:
        return tokenizer.cls_token_id
    return tokenizer.pad_token_id

def compute_token_accuracy(logits: Tensor, targets: Tensor, pad_idx: int) -> float:
    preds = torch.argmax(logits, dim=-1)
    mask = targets != pad_idx
    correct = (preds == targets) & mask
    return correct.sum().float() / max(1.0, mask.sum().float())

def compute_perplexity(loss: float) -> float:
    return math.exp(min(loss, 100.0))

def levenshtein_distance(s1: str, s2: str) -> int:
    if len(s1) < len(s2): return levenshtein_distance(s2, s1)
    if len(s2) == 0: return len(s1)
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
    total_dist = 0
    total_len = 0
    for p, t in zip(preds, targets):
        dist = levenshtein_distance(p, t)
        total_dist += dist
        total_len += len(t)
    return total_dist / max(1, total_len)

# Note: DeepSpeed handles gradient norm computation internally if configured, 
# but for manual logging we might still want to inspect it if needed.
# For simplicity and speed in multi-gpu, we might skip frequent full-model grad norm syncs
# or let DeepSpeed handle it status reporting.

# ============================================
# ARGS AND MAIN
# ============================================

def parse_args():
    parser = argparse.ArgumentParser(description="Train Spiking Transformer OCR")
    
    # Distributed args
    parser.add_argument("--local_rank", type=int, default=-1, help="local rank for distributed training")

    # Training config
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (per GPU)")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    
    # Model config
    parser.add_argument("--mask_mode", type=str, default="multiply", choices=["multiply", "additive"],
                        help="Masking mode for attention")
    parser.add_argument("--use_mssa", action="store_true", help="Use Multi-Scale Spiking Attention")
    parser.add_argument("--mssa_scales", type=str, default="1,2,4", help="Comma-separated MSSA scales")
    parser.add_argument("--in_channels", type=int, default=3, choices=[1, 3], help="Input channels")
    parser.add_argument("--num_steps", type=int, default=8, help="Number of SNN timesteps")
    
    # Image config
    parser.add_argument("--img_height", type=int, default=64, help="Image height")
    parser.add_argument("--img_width", type=int, default=768, help="Image width")
    
    # Curriculum learning
    parser.add_argument("--use_curriculum", action="store_true", help="Enable curriculum learning")
    
    # Optimizer
    parser.add_argument("--flora", action="store_true", help="Use Flora memory-efficient optimizer")

    # WandB logging
    parser.add_argument("--use_wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--wandb_project", type=str, default="spikeformer-ocr", help="WandB project name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="WandB run name")
    
    return parser.parse_args()


def train():
    args = parse_args()
    
    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank == -1 and args.local_rank != -1:
        local_rank = args.local_rank
        
    if local_rank != -1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        # Fallback for single GPU/CPU debugging
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    config = get_training_config(args)
    # Important: Update config device if needed, though mostly handled by DS/torch
    
    # ============================================
    # HYPERPARAMETERS
    # ============================================
    EMB_SIZE = 384
    NHEAD = 6
    FFN_HID_DIM = 4 * EMB_SIZE
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    NUM_STEPS = args.num_steps
    LR = args.lr
    BATCH_SIZE = config["batch_size"]
    NUM_EPOCHS = args.epochs
    IMG_SIZE = (args.img_height, args.img_width)
    LOG_EVERY = 100
    EVAL_EVERY = 2000
    LOG_PRINT_EVERY = 1000
    MAX_CHARS = 128 
    WEIGHT_DECAY = 0.01
    MASK_MODE = args.mask_mode
    USE_MSSA = args.use_mssa
    MSSA_SCALES = [int(x) for x in args.mssa_scales.split(",")]
    IN_CHANNELS = args.in_channels
    USE_CURRICULUM = args.use_curriculum

    if USE_MSSA and NHEAD % len(MSSA_SCALES) != 0:
        NHEAD = len(MSSA_SCALES) * (NHEAD // len(MSSA_SCALES) + 1)
        # Only print on rank 0
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print(f"[MSSA] Adjusted NHEAD to {NHEAD}")

    # Get available fonts count
    valid_fonts = get_font_pool()
    NUM_FONTS = len(valid_fonts)


    # WandB Config
    wandb_config = {
        "emb_size": EMB_SIZE,
        "n_heads": NHEAD,
        "num_encoder_layers": NUM_ENCODER_LAYERS,
        "learning_rate": LR,
        "batch_size": BATCH_SIZE,
        "img_size": IMG_SIZE,
        "deepspeed": False,
        "fsdp": True,
        "num_fonts": NUM_FONTS
    }
    
    # Init WandB helper (Does not login yet)
    init_wandb(args, wandb_config)
    
    # Login only on main process, after distributed init
    if (not dist.is_initialized()) or dist.get_rank() == 0:
        do_wandb_init_on_rank0(args, wandb_config)
        print("="*60)
        print("Configuration:")
        print(f"  Batch size (per GPU): {BATCH_SIZE}")
        print(f"  Learning rate: {LR}")
        print(f"  Image size: {IMG_SIZE}")
        print(f"  FSDP Enabled")
        print(f"  Fonts loaded: {NUM_FONTS}")
        print("="*60 + "\n")

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    PAD_IDX = processor.tokenizer.pad_token_id
    EOS_IDX = processor.tokenizer.eos_token_id
    START_IDX = get_decoder_start_id(processor.tokenizer)
    VOCAB_SIZE = len(processor.tokenizer)

    # ============================================
    # DATASETS
    # ============================================
    # Use 'worker_init_fn' logic or set seed effectively in dataset
    train_ds = WikiTextImageDataset(
        processor=processor,
        split="train",
        img_size=IMG_SIZE,
        train=True,
        max_samples=50_000_000,
        max_chars=MAX_CHARS,
        cache_size=10000, # Increased for IO buffering (was 100)
        in_channels=IN_CHANNELS,
        sources=[("wikimedia/wikipedia", "20231101.fr", "Français")],
    )

    val_ds = WikiTextImageDataset(
        processor=processor,
        split="test",
        img_size=IMG_SIZE,
        train=False,
        max_samples=10_000,
        max_chars=MAX_CHARS,
        in_channels=IN_CHANNELS,
        sources=[("wikimedia/wikipedia", "20231101.fr", "Français")],
    )

    data_collator = WikiTextDataCollator(processor, max_length=MAX_CHARS)

    # Reduce workers to avoid CPU saturation on multi-GPU
    # 4 workers per GPU * 2 GPUs = 8 workers total, leaving room for main processes
    num_workers = 4 

    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=num_workers, 
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collator,
        shuffle=False # Shuffle handled by dataset or distributed sampler
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=BATCH_SIZE, 
        num_workers=num_workers, 
        prefetch_factor=2,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=data_collator
    )

    # ============================================
    # MODEL
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
        mask_mode=MASK_MODE,
        use_mssa=USE_MSSA,
        mssa_scales=MSSA_SCALES,
        in_channels=IN_CHANNELS,
        img_height=IMG_SIZE[0],
    ).to(device)

    # Wrap with FSDP
    # Standard mixed precision policy for FSDP (BF16 if supported)
    bf16_ready = (
        torch.version.cuda
        and torch.cuda.is_bf16_supported()
        and os.environ.get("ACCELERATE_MIXED_PRECISION", "bf16") == "bf16"
    )

    mp_policy = None
    if bf16_ready:
        mp_policy = MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        )
    else:
        # Fallback to fp16
        mp_policy = MixedPrecision(
            param_dtype=torch.float16,
            reduce_dtype=torch.float16,
            buffer_dtype=torch.float16,
        )

    # Auto wrap policy: wrap layers > 10M params roughly, or transformer blocks
    my_auto_wrap_policy = size_based_auto_wrap_policy 

    model = FSDP(
        model,
        auto_wrap_policy=my_auto_wrap_policy,
        mixed_precision=mp_policy,
        device_id=device,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE
    )

    # ============================================
    # OPTIMIZER & SCHEDULER
    # ============================================
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)
    
    if args.flora:
        if (not dist.is_initialized()) or dist.get_rank() == 0:
            print("[Optimizer] Using Flora (Memory Efficient) Optimizer")
        try:
            from flora_opt import Flora
            optimizer = Flora(
                model.parameters(), 
                lr=LR, 
                weight_decay=WEIGHT_DECAY
            )
        except ImportError:
            raise ImportError("Flora optimizer requested but not installed. Run: pip install flora-opt")
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=LR, 
            betas=(0.9, 0.98),
            eps=1e-9,
            weight_decay=WEIGHT_DECAY
        )
    
    total_steps = NUM_EPOCHS * len(train_loader)
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=total_steps,
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1e4,
        # Flora (and some other custom optimizers) don't support momentum cycling
        cycle_momentum=(not args.flora)
    )

    model_engine = model # For compatibility with rest of code naming

    # ============================================
    # HELPERS
    # ============================================
    def step_loss(logits_steps: Tensor, tgt_out: Tensor) -> Tensor:
        # Cross entropy over timesteps (average)
        loss = 0.0
        for s in range(logits_steps.shape[0]):
            logits = logits_steps[s]
            loss = loss + criterion(
                logits.reshape(-1, logits.size(-1)), 
                tgt_out.reshape(-1)
            )
        return loss / logits_steps.shape[0]

    @torch.no_grad()
    def evaluate():
        model_engine.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        eval_cer = 0.0
        n_batches = 0
        
        # Only show pbar on rank 0
        is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
        disable_pbar = not is_main_process
        eval_pbar = tqdm(val_loader, desc="  Evaluation  ", unit="batch", disable=disable_pbar)
        
        for batch in eval_pbar:
            # device is already handled by loader/model if we move batch
            src = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            tgt_in = labels[:, :-1]
            tgt_out = labels[:, 1:]
            dec_mask = create_decoder_mask(tgt_in, PAD_IDX, device)

            # FSDP Mixed Precision handles casting for parameters; inputs might need cast if strict?
            # Usually torch handles float32 inputs -> BF16 operations automatically if autocast enabled or FSDP mixed precision
            # But FSDP mixed precision is primarily for weights/gradients. 
            # We don't need manual bfloat16 cast for inputs typically with FSDP mixed precision unless we want to save transfer bandwidth.
            
            # Forward
            logits_steps, _ = model_engine(
                src, tgt_in, 
                enc_padding_mask=None, 
                look_ahead_mask=dec_mask, 
                dec_padding_mask=None
            )
            loss = step_loss(logits_steps, tgt_out)
            
            mean_logits = logits_steps.mean(dim=0)
            acc = compute_token_accuracy(mean_logits, tgt_out, PAD_IDX)
            
            # Simple greedy decode for metric estimate
            pred_strs = strings_from_logits_until_eos(processor, mean_logits, EOS_IDX)
            tgt_strs = tokens_to_strings_until_eos(processor, tgt_out, EOS_IDX)
            cer = compute_cer(pred_strs, tgt_strs)
            
            eval_loss += loss.item()
            eval_acc += acc.item()
            eval_cer += cer
            n_batches += 1
            
            if not disable_pbar:
                eval_pbar.set_postfix({"L": f"{loss.item():.3f}", "A": f"{acc.item():.1%}"})
        
        # Dist reduce metrics
        metrics_tensor = torch.tensor([eval_loss, eval_acc, eval_cer, n_batches], device=device)
        
        if dist.is_initialized():
            dist.all_reduce(metrics_tensor, op=dist.ReduceOp.SUM)
        
        total_loss = metrics_tensor[0].item()
        total_acc = metrics_tensor[1].item()
        total_cer = metrics_tensor[2].item()
        total_batches = metrics_tensor[3].item()
        
        avg_loss = total_loss / max(1, total_batches)
        avg_acc = total_acc / max(1, total_batches)
        avg_cer = total_cer / max(1, total_batches)
        avg_ppl = compute_perplexity(avg_loss)
        
        cur_lr = optimizer.param_groups[0]['lr']
        
        return avg_loss, avg_acc, avg_cer, avg_ppl, cur_lr

    @torch.no_grad()
    def print_examples_rank0(batch_src, batch_labels, logits_steps):
        if dist.is_initialized() and dist.get_rank() != 0: return
        
        logits = logits_steps.mean(dim=0)
        pred_tf_str = strings_from_logits_until_eos(processor, logits, EOS_IDX)
        gt_str = tokens_to_strings_until_eos(processor, batch_labels, EOS_IDX)

        print("\n" + "="*60)
        print("Examples (Teacher Forcing Only):") # Greedy decode might be tricky with distributed engine wrappers
        print("="*60)
        for i in range(min(3, batch_src.size(0))):
            print(f"  GT: {gt_str[i]}")
            print(f"  TF: {pred_tf_str[i]}")
        print("="*60 + "\n", flush=True)

    # ============================================
    # TRAINING LOOP
    # ============================================
    global_step = 0
    best_val = float("inf")
    
    # Resume checkpoint if needed
    # (FSDP loading checkpoint logic would go here, simplified for now: manual load on rank 0 or FSDP load)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model_engine.train()
        
        # Only rank 0 bar
        is_main_process = (not dist.is_initialized()) or (dist.get_rank() == 0)
        disable_pbar = not is_main_process
        if is_main_process:
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}", unit="batch", disable=disable_pbar)
        else:
            pbar = train_loader
        
        running_loss = 0.0
        
        for batch in pbar:
            global_step += 1
            
            src = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            tgt_in = labels[:, :-1]
            tgt_out = labels[:, 1:]
            dec_mask = create_decoder_mask(tgt_in, PAD_IDX, device)
            
            # Forward
            logits_steps, _ = model_engine(
                src, tgt_in, 
                enc_padding_mask=None, 
                look_ahead_mask=dec_mask, 
                dec_padding_mask=None
            )
            loss = step_loss(logits_steps, tgt_out)

            # Backward
            loss.backward()
            
            # Gradient Clipping
            model_engine.clip_grad_norm_(1.0)
            
            optimizer.step()
            if scheduler: scheduler.step()
            optimizer.zero_grad()
            
            # Logging
            loss_val = loss.item()
            running_loss += loss_val
            
            if is_main_process:
                if global_step % 10 == 0:
                    lr_val = optimizer.param_groups[0]['lr']
                        
                    log_metrics({
                        "train/loss": loss_val,
                        "train/perplexity": compute_perplexity(loss_val),
                        "train/lr": lr_val,
                        "train/epoch": epoch,
                    }, step=global_step)
            
                if global_step % LOG_EVERY == 0:
                    avg_l = running_loss / LOG_EVERY
                    running_loss = 0.0
                    pbar.set_postfix({"Loss": f"{avg_l:.4f}"})
            
            if global_step % EVAL_EVERY == 0:
                val_loss, val_acc, val_cer, val_ppl, cur_lr = evaluate()
                model_engine.train()
                
                if is_main_process:
                    print(f"\nEvaluation: Loss={val_loss:.4f} CER={val_cer:.2%}")
                    log_metrics({
                        "val/loss": val_loss,
                        "val/cer": val_cer,
                        "val/acc": val_acc
                    }, step=global_step)
                    
                    if val_loss < best_val:
                        best_val = val_loss
                        # Save checkpoint
                        # Simple FSDP State Dictionary saving (Full State Dict)
                        # Warning: This gathers all weights to CPU/Rank 0.
                        # For very large models, use SHARDED_STATE_DICT.
                        from torch.distributed.fsdp import StateDictType, FullStateDictConfig
                        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
                        with FSDP.state_dict_type(model_engine, StateDictType.FULL_STATE_DICT, save_policy):
                            cpu_state = model_engine.state_dict()
                            if dist.get_rank() == 0:
                                torch.save(cpu_state, f"checkpoints/spikeformer_best.pt")
                        
                        if dist.get_rank() == 0:
                            print("Saved best checkpoint.")
        
    finish_wandb()

if __name__ == "__main__":
    train()