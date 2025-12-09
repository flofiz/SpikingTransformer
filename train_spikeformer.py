import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Pour éviter les warnings inutiles
import math
import time
from typing import Tuple, List
from tqdm.auto import tqdm
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

from transformers import TrOCRProcessor
import torch.nn.functional as F
from Triton_Layers.Seq2Seq import Seq2Seq as Seq2SeqTransformer
from wiki_text_images3 import WikiTextImageDataset, WikiTextDataCollator


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


def train():
    # ============================================
    # HYPERPARAMETERS - Optimisés
    # ============================================
    EMB_SIZE = 384
    NHEAD = 6
    FFN_HID_DIM = 4 * EMB_SIZE
    NUM_ENCODER_LAYERS = 6
    NUM_DECODER_LAYERS = 6
    NUM_STEPS = 4
    LR = 1e-3
    BATCH_SIZE = 96
    NUM_EPOCHS = 1
    IMG_SIZE = (32, 512)  # H, W pour le resize/pad du générateur
    LOG_EVERY = 100
    EVAL_EVERY = 2000
    LOG_PRINT_EVERY = 1000
    MAX_CHARS = 128 
    GRAD_CLIP_NORM = 1.0  # ✅ Gradient clipping
    WEIGHT_DECAY = 0.01   # ✅ Régularisation

    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
    PAD_IDX = processor.tokenizer.pad_token_id
    EOS_IDX = processor.tokenizer.eos_token_id
    START_IDX = get_decoder_start_id(processor.tokenizer)
    VOCAB_SIZE = len(processor.tokenizer)

    # ============================================
    # DATASETS
    # ============================================
    # train_ds = WikiTextImageDataset(
    #     processor=processor,
    #     split="train",
    #     img_size=IMG_SIZE,
    #     languages=["20220301.en", "20220301.fr", "20220301.de"],
    #     train=True,
    #     max_samples=50_000_000,
    #     max_chars=MAX_CHARS,
    # )
    # val_ds = WikiTextImageDataset(
    #     processor=processor,
    #     split="test",
    #     img_size=IMG_SIZE,
    #     train=False,
    #     max_samples=2_000,
    #     max_chars=MAX_CHARS,
    #     languages=["20220301.en", "20220301.fr", "20220301.de"],
    # )

    # train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=16, prefetch_factor=4,          # Précharge 4 batches par worker (32 batches total!)
    # persistent_workers=True,     # Important pour le streaming
    # pin_memory=True,            # Si vous utilisez un GPU
    # drop_last=True,)
    # val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=16, prefetch_factor=4,          # Précharge 4 batches par worker (32 batches total!)
    # persistent_workers=True,     # Important pour le streaming
    # pin_memory=True,            # Si vous utilisez un GPU
    # drop_last=True,)

    train_ds = WikiTextImageDataset(
        processor=processor,
        split="train",
        img_size=IMG_SIZE,
        train=True,
        max_samples=50_000_000,
        max_chars=MAX_CHARS,
        cache_size=100,
        article_rotation_interval=500_000,
        # lang=[
        #     "20231101.en",  # Anglais
        #     "20231101.fr",  # Français
        #     "20231101.de",  # Allemand
        #     "20231101.es",  # Espagnol
        #     "20231101.it",  # Italien
        #     "20231101.pt",  # Portugais
        #     "20231101.nl",  # Néerlandais
        # ],
        # sources=[
        #     ("HuggingFaceFW/clean-wikipedia", "fr", "Français"),
        #     ("HuggingFaceFW/clean-wikipedia", "en", "Anglais"),
        #     ("HuggingFaceFW/clean-wikipedia", "de", "Allemand"),
        #     ("HuggingFaceFW/clean-wikipedia", "es", "Espagnol"),
        #     ("HuggingFaceFW/clean-wikipedia", "it", "Italien"),
        #     ("HuggingFaceFW/clean-wikipedia", "pt", "Portugais"),
        #     ("HuggingFaceFW/clean-wikipedia", "nl", "Néerlandais"),
        # ],
        sources=[
            ("wikimedia/wikipedia", "20231101.fr", "Français"),
        ],
        # enable_all_wikipedia=True,  # Active toutes les 18 langues latines
    )

    val_ds = WikiTextImageDataset(
        processor=processor,
        split="test",
        img_size=IMG_SIZE,
        train=False,
        max_samples=10_000,
        max_chars=MAX_CHARS,
        sources=[
            # ("HuggingFaceFW/clean-wikipedia", "fr", "Français"),
            ("wikimedia/wikipedia", "20231101.fr", "Français"),
        ]
    )

    # ✅ Data Collator pour padding dynamique
    data_collator = WikiTextDataCollator(processor, max_length=MAX_CHARS)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, num_workers=8, prefetch_factor=2,          # Précharge 2 batches par worker (16 batches total!)
    persistent_workers=True,     # Important pour le streaming
    pin_memory=True,            # Si vous utilisez un GPU
    drop_last=True,
    collate_fn=data_collator)    # ✅ Ajout du collator

    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, num_workers=8, prefetch_factor=2,          # Précharge 2 batches par worker (16 batches total!)
    persistent_workers=True,     # Important pour le streaming
    pin_memory=True,            # Si vous utilisez un GPU
    drop_last=True,
    collate_fn=data_collator)    # ✅ Ajout du collator

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
    ).to(DEVICE)

    # ============================================
    # LOSS, OPTIMIZER, SCHEDULER - Optimisés
    # ============================================
    # ✅ CrossEntropyLoss gère déjà log_softmax en interne (fused, plus rapide)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.1)  # ✅ Label smoothing
    
    # ✅ Betas optimisés pour transformers + weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR, 
        betas=(0.9, 0.98),  # ✅ Changé de (0.9, 0.99) à (0.9, 0.98)
        eps=1e-9,
        weight_decay=WEIGHT_DECAY  # ✅ Régularisation L2
    )
    
    scaler = GradScaler()
    total_steps = NUM_EPOCHS * len(train_loader)
    
    # ✅ OneCycleLR avec warmup intégré - optimal pour 1 epoch
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=total_steps,
        pct_start=0.1,        # 10% de warmup
        anneal_strategy='cos', # Cosine annealing
        div_factor=25,        # Start LR = max_lr/25 = 4e-5
        final_div_factor=1e4  # End LR = max_lr/1e4 = 1e-7
    )
    
    # Alternative pour multi-epochs (décommenter si NUM_EPOCHS > 1):
    # warmup_steps = total_steps // 10
    # scheduler1 = torch.optim.lr_scheduler.LinearLR(
    #     optimizer, start_factor=0.01, total_iters=warmup_steps
    # )
    # scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=total_steps - warmup_steps, eta_min=1e-6
    # )
    # scheduler = torch.optim.lr_scheduler.SequentialLR(
    #     optimizer, schedulers=[scheduler1, scheduler2], milestones=[warmup_steps]
    # )

    # ============================================
    # LOSS FUNCTION - Corrigée
    # ============================================
    def step_loss(logits_steps: Tensor, tgt_out: Tensor) -> Tensor:
        """
        ✅ CORRECTION MAJEURE: Pas de log_softmax !
        CrossEntropyLoss attend des logits bruts et applique log_softmax en interne.
        """
        loss = 0.0

        # print(logits_steps.shape, tgt_out.shape, flush=True)
        for s in range(logits_steps.shape[0]):
            logits = logits_steps[s]  # (B, T, V) - LOGITS BRUTS
            loss = loss + criterion(
                logits.reshape(-1, logits.size(-1)), 
                tgt_out.reshape(-1)
            )
        return loss / logits_steps.shape[0]

    # ============================================
    # EVALUATION FUNCTION
    # ============================================
    @torch.no_grad()
    def evaluate() -> Tuple[float, float, float, float]:
        model.eval()
        eval_loss = 0.0
        eval_acc = 0.0
        eval_cer = 0.0
        n_batches = 0
        
        eval_pbar = tqdm(val_loader, desc="  Evaluation  ", unit="batch")
        for batch in eval_pbar:
            src = batch["pixel_values"].to(DEVICE, non_blocking=True)
            labels = batch["labels"].to(DEVICE, non_blocking=True)
            tgt_in = labels[:, :-1]
            tgt_out = labels[:, 1:]
            dec_mask = create_decoder_mask(tgt_in, PAD_IDX, DEVICE)
            
            with autocast(dtype=torch.float16):
                logits_steps, _ = model(
                    src, tgt_in, 
                    enc_padding_mask=None, 
                    look_ahead_mask=dec_mask, 
                    dec_padding_mask=None
                )
                # ✅ Pas de log_softmax ici !
                loss = step_loss(logits_steps, tgt_out)
            
            # --- Metrics ---
            mean_logits = logits_steps.mean(dim=0)
            
            # 1. Token Accuracy
            acc = compute_token_accuracy(mean_logits, tgt_out, PAD_IDX)
            
            # 2. CER (Character Error Rate)
            # Décodage pour CER (coûteux, donc fait sur batch complet ou sous-échantillon)
            # On le fait sur tout le batch pour précision
            pred_strs = strings_from_logits_until_eos(processor, mean_logits, EOS_IDX)
            tgt_strs = tokens_to_strings_until_eos(processor, tgt_out, EOS_IDX)
            cer = compute_cer(pred_strs, tgt_strs)
            
            eval_loss += loss.item()
            eval_acc += acc.item()
            eval_cer += cer
            n_batches += 1
            
            # Update pbar
            eval_pbar.set_postfix({"L": f"{loss.item():.3f}", "A": f"{acc.item():.1%}", "C": f"{cer:.1%}"})
        
        avg_loss = eval_loss / max(1, n_batches)
        avg_acc = eval_acc / max(1, n_batches)
        avg_cer = eval_cer / max(1, n_batches)
        
        return avg_loss, avg_acc, avg_cer, scheduler.get_last_lr()[0]

    # ============================================
    # PRINT EXAMPLES FUNCTION
    # ============================================
    @torch.no_grad()
    def print_examples(batch_src, batch_labels, logits_steps, max_examples=3):
        # Moyenne sur les steps (SNN)
        logits = logits_steps.mean(dim=0)  # (B,T,V)

        # Pred teacher-forcing (masqué)
        pred_tf_str = strings_from_logits_until_eos(processor, logits, EOS_IDX)

        # Ground-truth tronquée à EOS
        gt_str = tokens_to_strings_until_eos(processor, batch_labels, EOS_IDX)

        print("\n" + "="*60)
        print("Exemples (Teacher Forcing vs Greedy Decoding):")
        print("="*60)
        nb = min(max_examples, batch_src.size(0))
        for i in range(nb):
            # Génération greedy
            ys, _, _ = model.greedy_decode(
                src=batch_src[i],
                max_len=batch_labels.size(1),
                start_symbol=START_IDX,
                eos_idx=EOS_IDX,
                pad_idx=PAD_IDX,
                device=DEVICE,
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
            src0 = first_batch["pixel_values"].to(DEVICE, non_blocking=True)[0]  # (3,H,W)
            labels0 = first_batch["labels"].to(DEVICE, non_blocking=True)[0]     # (T,)
            _ys, _attn, _out = model.greedy_decode(
                src=src0,
                max_len=labels0.size(0),
                start_symbol=START_IDX,
                eos_idx=EOS_IDX,
                pad_idx=PAD_IDX,
                device=DEVICE,
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
    # Listez tous les paramètres
    model_params = set(model.parameters())
    optimizer_params = set()
    for group in optimizer.param_groups:
        optimizer_params.update(group['params'])

    missing = model_params - optimizer_params
    if missing:
        print(f"❌ {len(missing)} paramètres ne sont PAS dans l'optimizer!")
        for name, param in model.named_parameters():
            if param in missing:
                print(f"  - {name}")
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
        
        checkpoint = torch.load(ckpt_path, map_location=DEVICE)
        
        # Charger le modèle
        msg = model.load_state_dict(checkpoint["model"], strict=False)
        print(f"Model loaded: {msg}")
        
        # Charger l'optimizer
        if "optimizer" in checkpoint and checkpoint["optimizer"] is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
            print("Optimizer loaded")
            
        # Charger le scheduler
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
            print("Scheduler loaded")
            
        # Charger les scalaires
        global_step = checkpoint.get("step", 0)
        best_val = checkpoint.get("val_loss", float("inf"))
        
        # Estimer l'epoch de départ (approximatif)
        steps_per_epoch = len(train_loader)
        if steps_per_epoch > 0:
            start_epoch = (global_step // steps_per_epoch) + 1
            
        print(f"Make sure to adjust LR scheduling if needed.")
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
            optimizer.zero_grad(set_to_none=True)

            src = batch["pixel_values"].to(DEVICE, non_blocking=True)  # (B,3,H,W)
            labels = batch["labels"].to(DEVICE, non_blocking=True)     # (B,T)
            tgt_in = labels[:, :-1]
            tgt_out = labels[:, 1:]

            dec_mask = create_decoder_mask(tgt_in, PAD_IDX, DEVICE)

            with autocast(dtype=torch.float16):
                logits_steps, _ = model(
                    src, tgt_in, 
                    enc_padding_mask=None, 
                    look_ahead_mask=dec_mask, 
                    dec_padding_mask=None
                )
                # ✅ CORRECTION: Suppression du log_softmax
                # logits_steps reste des logits bruts
                loss = step_loss(logits_steps, tgt_out)
                
                # Metric: Token accuracy (sur le dernier step ou moyenne des logits)
                # On utilise la moyenne des steps pour la décision finale (comme en inférence)
                mean_logits = logits_steps.mean(dim=0)
                acc = compute_token_accuracy(mean_logits, tgt_out, PAD_IDX)

            scaler.scale(loss).backward()
            # print("\n=== GRADIENT NORMS ===")
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         grad_norm = param.grad.norm().item()
            #         param_norm = param.norm().item()
            #         print(f"{name:40s} | grad: {grad_norm:.6e} | param: {param_norm:.6e}")
            #     else:
            #         print(f"{name:40s} | grad: NONE")
            
            # ✅ Gradient clipping pour stabilité (important pour SNN)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # ✅ Step après chaque batch (requis pour OneCycleLR)
        
            running_loss += loss.item()
            running_acc += acc.item()

            # Logging régulier
            if global_step % LOG_EVERY == 0:
                avg_loss = running_loss / LOG_EVERY
                avg_acc = running_acc / LOG_EVERY
                running_loss = 0.0
                running_acc = 0.0
                pbar.set_postfix({
                    "Loss": f"{avg_loss:.4f}", 
                    "Acc": f"{avg_acc:.2%}",
                    "LR": f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
            # Print examples
            if global_step % LOG_PRINT_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    print_examples(src, labels, logits_steps, max_examples=3)
                model.train()

            # Evaluation périodique
            if global_step % EVAL_EVERY == 0:
                val_loss, val_acc, val_cer, cur_lr = evaluate()
                print(f"\n{'='*60}")
                print(f"📊 Evaluation @ step {global_step}/{total_steps}")
                print(f"   Val Loss: {val_loss:.4f}")
                print(f"   Val Acc : {val_acc:.2%}")
                print(f"   Val CER : {val_cer:.2%}")
                print(f"   Learning Rate: {cur_lr:.2e}")
                print(f"{'='*60}\n", flush=True)
                
                if val_loss < best_val:
                    best_val = val_loss
                    os.makedirs("checkpoints", exist_ok=True)
                    ckpt_path = "checkpoints/spikeformer2_best.pt"
                    torch.save({
                        "model": model.state_dict(), 
                        "optimizer": optimizer.state_dict(), 
                        "scheduler": scheduler.state_dict(),  # ✅ Sauvegarde du scheduler
                        "step": global_step,
                        "step": global_step,
                        "val_loss": val_loss,
                        "val_acc": val_acc,
                        "val_cer": val_cer,
                        "config": {
                            "emb_size": EMB_SIZE,
                            "nhead": NHEAD,
                            "num_encoder_layers": NUM_ENCODER_LAYERS,
                            "num_decoder_layers": NUM_DECODER_LAYERS,
                            "vocab_size": VOCAB_SIZE,
                        }
                    }, ckpt_path)
                    print(f"✅ Saved best checkpoint to {ckpt_path} (val_loss: {val_loss:.4f})\n")
                
                model.train()

        epoch_time = time.time() - t0
        print(f"\n{'='*60}")
        print(f"Epoch {epoch} completed in {epoch_time/60:.1f} min")
        print(f"{'='*60}\n", flush=True)

    print("="*60)
    print("✅ Training completed successfully!")
    print(f"Best validation loss: {best_val:.4f}")
    print("="*60, flush=True)


if __name__ == "__main__":
    train()