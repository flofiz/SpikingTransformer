import os
import re
import random
import logging
from typing import List, Optional, Tuple, Union, Dict, Any
from itertools import cycle
from threading import Lock

import torch
from torch.utils.data import IterableDataset
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from torchvision import transforms
from datasets import load_dataset
import torchvision.transforms.functional as TF
import torch.nn.functional as F


# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _find_font(candidates: Optional[List[str]] = None, size: int = 28) -> ImageFont.FreeTypeFont:
    """Trouve une police disponible sur le système."""
    if candidates is None:
        candidates = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
            "/Library/Fonts/Arial.ttf",
        ]
    for fp in candidates:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def _clean_and_split_sentences(
    text: str, 
    min_chars: int = 10, 
    max_chars: int = 120,
    max_overlap: float = 0.25
) -> List[str]:
    """
    Split optimisé du texte pour maximiser le nombre d'échantillons diversifiés.
    
    Args:
        text: Texte source à découper
        min_chars: Longueur minimale des segments
        max_chars: Longueur maximale des segments
        max_overlap: Overlap maximal pour les fenêtres glissantes (0.0 à 1.0)
    
    Returns:
        Liste de segments de texte uniques
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    
    out = []
    seen = set()  # Évite les doublons en O(1)
    
    def add_unique(segment: str):
        """Ajoute un segment s'il est unique."""
        segment = segment.strip()
        if min_chars <= len(segment) <= max_chars and segment not in seen:
            seen.add(segment)
            out.append(segment)
    
    # Stratégie 1: Split par phrases (ponctuation forte)
    sentences = re.split(r"[.!?;]\s+", text)
    for s in sentences:
        s = s.strip()
        s = re.sub(r"\[[^\]]*\]", "", s)  # Supprime [références]
        s = re.sub(r"\([^)]*\)", "", s)    # Supprime (parenthèses)
        s = s.strip()
        
        if min_chars <= len(s) <= max_chars:
            add_unique(s)
        
        # Stratégie 2: Split par virgules et conjonctions (pour phrases longues)
        elif len(s) > max_chars:
            sub_segments = re.split(
                r",\s+|:\s+|\s+et\s+|\s+ou\s+|\s+mais\s+|\s+donc\s+|\s+car\s+", 
                s
            )
            for sub in sub_segments:
                add_unique(sub)
                
                # Stratégie 3: Fenêtre glissante avec overlap contrôlé
                if len(sub) > max_chars:
                    words = sub.split()
                    if len(words) >= 3:
                        window_size = max(3, len(words) // 3)
                        step = max(1, int(window_size * (1 - max_overlap)))
                        
                        for i in range(0, len(words) - window_size + 1, step):
                            segment = " ".join(words[i:i + window_size])
                            add_unique(segment)
    
    # Stratégie 4: Chunks avec overlap réduit
    words = text.split()
    if len(words) >= 5:
        chunk_words = max(3, max_chars // 8)  # ~8 chars/mot
        step = max(1, int(chunk_words * (1 - max_overlap)))
        
        for i in range(0, len(words) - chunk_words + 1, step):
            chunk = " ".join(words[i:i + chunk_words])
            chunk = re.sub(r"\[[^\]]*\]", "", chunk)
            chunk = re.sub(r"\([^)]*\)", "", chunk)
            add_unique(chunk)
    
    return out


def _draw_text_to_image(
    text: str,
    img_size: Tuple[int, int] = (384, 384),
    margin: int = 7,
    base_font_size: int = 64,
    font_candidates: Optional[List[str]] = None,
    jitter_font: Tuple[int, int] = (-6, 6),
    train: bool = True,
) -> Image.Image:
    """Génère une image à partir de texte."""
    bg_color = 255
    fg_color = 0

    font_size = base_font_size
    font = _find_font(font_candidates, size=font_size)

    line_text = " ".join(text.split())

    # Optimisation: utilise font.getbbox si disponible (PIL >= 8.0.0)
    try:
        bbox = font.getbbox(line_text) if line_text else (0, 0, 0, 0)
    except AttributeError:
        # Fallback pour anciennes versions
        tmp_img = Image.new("L", (1, 1), color=bg_color)
        tmp_draw = ImageDraw.Draw(tmp_img)
        bbox = tmp_draw.textbbox((0, 0), line_text, font=font) if line_text else (0, 0, 0, 0)
    
    text_w = max(0, bbox[2] - bbox[0])
    text_h = max(0, bbox[3] - bbox[1])

    if text_w == 0 or text_h == 0:
        try:
            ascent, descent = font.getmetrics()
        except Exception:
            ascent, descent = (font_size, int(font_size * 0.25))
        text_h = max(text_h, ascent + descent)
        try:
            text_w = max(text_w, int(font.getlength(" ")))
        except Exception:
            text_w = max(text_w, max(1, font_size // 2))

    W = int(text_w + 2 * margin)
    H = int(text_h + 2 * margin)

    img = Image.new("L", (W, H), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((margin, margin), line_text, fill=fg_color, font=font)

    # Garde en niveaux de gris (pas de conversion RGB)
    return img


def pil_to_tensor_resize_pad(
    img: Image.Image,
    target_size: Union[int, Tuple[int, int]],
    pad_value: int = 0,
    normalize: bool = False,
    mean: Tuple[float, ...] = (0.5,),
    std: Tuple[float, ...] = (0.5,),
    resample=Image.BILINEAR,
    return_mask: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Redimensionne et pad une image en tensor.
    Optimisé pour images en niveaux de gris (1 canal).
    """
    if pad_value not in (0, 1):
        raise ValueError("pad_value doit être 0 ou 1")

    if isinstance(target_size, int):
        target_h = target_w = int(target_size)
    else:
        target_h, target_w = target_size

    # Force niveaux de gris
    if img.mode != "L":
        img = img.convert("L")

    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    resized = img.resize((new_w, new_h), resample=resample)

    tensor = TF.to_tensor(resized)  # Forme (1, new_h, new_w)
    mask = torch.ones((new_h, new_w), dtype=torch.float32)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    
    if pad_w < 0 or pad_h < 0:
        resized = resized.resize((target_w, target_h), resample=resample)
        tensor = TF.to_tensor(resized)
        mask = torch.ones((target_h, target_w), dtype=torch.float32)
    else:
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        
        # Padding: (left, right, top, bottom) pour F.pad
        pad_tuple = (pad_left, pad_right, pad_top, pad_bottom)
        
        if any(p > 0 for p in pad_tuple):
            tensor = F.pad(tensor, pad_tuple, mode="constant", value=float(pad_value))
            
            # Padding du masque avec la même convention
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            mask = F.pad(mask, pad_tuple, mode="constant", value=0.0)
            mask = mask.squeeze(0).squeeze(0)  # Retour à (H, W)

    if normalize:
        tensor = TF.normalize(tensor, mean=list(mean), std=list(std))

    return (tensor, mask) if return_mask else tensor


class TextSourceConfig:
    """Configuration des sources de texte disponibles."""
    
    # Wikipedia - Alphabets latins uniquement
    WIKIPEDIA_LATIN_SOURCES = [
        ("wikipedia", "20220301.fr", "Français"),
        ("wikipedia", "20220301.en", "Anglais"),
        ("wikipedia", "20220301.es", "Espagnol"),
        ("wikipedia", "20220301.de", "Allemand"),
        ("wikipedia", "20220301.it", "Italien"),
        ("wikipedia", "20220301.pt", "Portugais"),
        ("wikipedia", "20220301.nl", "Néerlandais"),
        ("wikipedia", "20220301.pl", "Polonais"),
        ("wikipedia", "20220301.tr", "Turc"),
        ("wikipedia", "20220301.sv", "Suédois"),
        ("wikipedia", "20220301.vi", "Vietnamien"),
        ("wikipedia", "20220301.ro", "Roumain"),
        ("wikipedia", "20220301.cs", "Tchèque"),
        ("wikipedia", "20220301.da", "Danois"),
        ("wikipedia", "20220301.fi", "Finnois"),
        ("wikipedia", "20220301.no", "Norvégien"),
        ("wikipedia", "20220301.hu", "Hongrois"),
        ("wikipedia", "20220301.id", "Indonésien"),
    ]
    
    # Autres sources - Alphabets latins uniquement
    OTHER_LATIN_SOURCES = [
        ("oscar", "unshuffled_deduplicated_fr", "OSCAR-FR"),
        ("oscar", "unshuffled_deduplicated_en", "OSCAR-EN"),
        ("oscar", "unshuffled_deduplicated_de", "OSCAR-DE"),
        ("oscar", "unshuffled_deduplicated_es", "OSCAR-ES"),
        ("oscar", "unshuffled_deduplicated_it", "OSCAR-IT"),
        ("c4", "en", "C4-EN"),
        ("mc4", "fr", "mC4-FR"),
        ("mc4", "en", "mC4-EN"),
        ("mc4", "es", "mC4-ES"),
        ("mc4", "de", "mC4-DE"),
        ("mc4", "it", "mC4-IT"),
        ("bookcorpus", None, "BookCorpus"),
    ]



class WikiTextDataCollator:
    """
    Collator qui gère la tokenisation dynamique et le padding.
    """
    def __init__(self, processor, max_length: int = 128):
        self.processor = processor
        self.max_length = max_length

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # 1. Stack des images et masques
        pixel_values = torch.stack([item["pixel_values"] for item in batch])
        pixel_mask = torch.stack([item["pixel_mask"] for item in batch])
        
        # 2. Récupération des textes bruts
        texts = [item["text_label"] for item in batch]
        
        # 3. Tokenisation dynamique (pad au max du batch)
        labels = self.processor.tokenizer(
            texts,
            padding=True,              # Pad à la séquence la plus longue du batch
            truncation=True,           # Tronque si dépasse max_length
            max_length=self.max_length,
            return_tensors="pt"
        ).input_ids
        
        return {
            "pixel_values": pixel_values,
            "pixel_mask": pixel_mask,
            "labels": labels,
        }


class MultiSourceTextDataset(IterableDataset):
    """
    Dataset infini multi-sources optimisé.
    - Support de multiples sources (Wikipedia multi-langues, OSCAR, C4, etc.)
    - Thread-safe pour DataLoader multi-workers
    - Rotation intelligente entre sources
    - Cache optimisé avec locks
    """
    
    def __init__(
        self,
        processor,
        split: str = "train",
        sources: Optional[List[Tuple[str, Optional[str], str]]] = None,
        languages: Optional[List[str]] = None,
        max_target_length: int = 128,
        img_size: Union[int, Tuple[int, int]] = (32, 512),
        min_chars: int = 10,
        max_chars: int = 100,
        max_overlap: float = 0.25,
        font_candidates: Optional[List[str]] = None,
        base_font_size: int = 28,
        train: bool = True,
        seed: int = 42,
        test_size: float = 0.02,
        cache_size: int = 50000,
        article_rotation_interval: int = 1000,
        enable_all_wikipedia: bool = False,
        enable_extended_sources: bool = False,
    ):
        super().__init__()
        assert split in ("train", "test")

        self.processor = processor
        self.max_target_length = max_target_length
        self.img_size = img_size
        self.font_candidates = font_candidates
        self.base_font_size = base_font_size
        self.train = train
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_overlap = max_overlap
        self.seed = seed
        self.split = split
        self.test_size = test_size
        self.cache_size = cache_size
        self.article_rotation_interval = article_rotation_interval

        # Configuration des sources
        if sources is None:
            sources = []
            
            if languages is not None:
                for lang_code in languages:
                    lang_name = lang_code.split(".")[-1].upper()
                    sources.append(("wikipedia", lang_code, f"Wikipedia-{lang_name}"))
            
            if not sources and not enable_all_wikipedia and not enable_extended_sources:
                sources = [
                    ("wikipedia", "20220301.fr", "Français"),
                    ("wikipedia", "20220301.en", "Anglais"),
                ]
            
            if enable_all_wikipedia:
                sources.extend(TextSourceConfig.WIKIPEDIA_LATIN_SOURCES)
            
            if enable_extended_sources:
                sources.extend(TextSourceConfig.OTHER_LATIN_SOURCES)
        
        self.sources = sources
        self.current_source_idx = 0
        self.articles_from_current_source = 0  # Compte les ARTICLES, pas les segments
        
        logger.info(f"[MultiSourceTextDataset] Initialisé avec {len(self.sources)} sources:")
        for i, (dataset, config, name) in enumerate(self.sources):
            logger.info(f"  [{i}] {name} ({dataset}/{config if config else 'default'})")
        logger.info(f"\n[MultiSourceTextDataset] Configuration:")
        logger.info(f"  - Split multi-stratégies avec overlap={max_overlap}")
        logger.info(f"  - Rotation toutes les {article_rotation_interval} articles")
        logger.info(f"  - Cache cible: {self.cache_size:,} segments\n")

        # Cache thread-safe
        self._sentence_cache = []
        self._cache_lock = Lock()
        self._current_iterator = None
        self._initialized = False
        self._failed_sources = set()

        # Augmentations
        self.post_color_aug = transforms.Compose([]) if not train else transforms.Compose([
            transforms.RandomApply(
                torch.nn.ModuleList([
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                ]),
                p=0.3,
            )
        ])

    def _get_next_source(self) -> Tuple[str, Optional[str], str]:
        """Obtient la prochaine source valide."""
        attempts = 0
        max_attempts = len(self.sources)
        
        while attempts < max_attempts:
            source_idx = self.current_source_idx % len(self.sources)
            
            if source_idx not in self._failed_sources:
                return self.sources[source_idx]
            
            self.current_source_idx += 1
            attempts += 1
        
        # Toutes les sources ont échoué
        logger.warning("[MultiSourceTextDataset] Toutes les sources échouées, réinitialisation...")
        self._failed_sources.clear()
        self.current_source_idx = 0
        return self.sources[0]

    def _initialize_stream(self):
        """Initialise le stream pour la source courante."""
        if self._initialized:
            return
        
        dataset_name, config_name, display_name = self._get_next_source()
        
        try:
            logger.info(f"[MultiSourceTextDataset] Chargement de {display_name}...")
            
            if config_name is not None:
                ds = load_dataset(
                    dataset_name, 
                    config_name, 
                    split="train", 
                    streaming=True, 
                    trust_remote_code=True
                )
            else:
                ds = load_dataset(
                    dataset_name, 
                    split="train", 
                    streaming=True, 
                    trust_remote_code=True
                )

            ds = ds.shuffle(seed=self.seed + self.current_source_idx, buffer_size=10000)
            
            if self.split == "test":
                n_test = int(self.test_size * 100000)
                ds = ds.take(n_test)
            else:
                n_test = int(self.test_size * 100000)
                ds = ds.skip(n_test)
            
            self._current_iterator = iter(ds)
            self._initialized = True
            self.articles_from_current_source = 0
            
            logger.info(f"[MultiSourceTextDataset] ✓ {display_name} chargé")
            
        except Exception as e:
            logger.error(f"[MultiSourceTextDataset] ✗ Erreur {display_name}: {e}")
            self._failed_sources.add(self.current_source_idx % len(self.sources))
            self.current_source_idx += 1
            self._initialized = False
            self._initialize_stream()

    def _rotate_source_if_needed(self):
        """Change de source si l'intervalle d'articles est atteint."""
        if self.articles_from_current_source >= self.article_rotation_interval:
            old_source = self.sources[self.current_source_idx % len(self.sources)][2]
            self.current_source_idx += 1
            new_source = self.sources[self.current_source_idx % len(self.sources)][2]
            logger.info(f"[MultiSourceTextDataset] Rotation: {old_source} → {new_source}")
            self._initialized = False
            self._current_iterator = None
            self.articles_from_current_source = 0

    def _fill_cache(self):
        """Remplit le cache avec de nouvelles phrases (thread-safe)."""
        if not self._initialized:
            self._initialize_stream()
        
        self._rotate_source_if_needed()
        
        articles_processed = 0
        max_articles_per_fill = 100
        
        while len(self._sentence_cache) < self.cache_size and articles_processed < max_articles_per_fill:
            try:
                ex = next(self._current_iterator)
                articles_processed += 1
                self.articles_from_current_source += 1
                
                # Extraction du texte
                txt = None
                for key in ["text", "content", "sentence", "paragraph"]:
                    if key in ex:
                        txt = ex[key]
                        break
                
                if not txt or len(txt) < self.min_chars:
                    continue
                
                # Split optimisé
                segs = _clean_and_split_sentences(
                    txt, 
                    self.min_chars, 
                    self.max_chars,
                    self.max_overlap
                )
                
                if len(segs) > 0:
                    self._sentence_cache.extend(segs)
                    
                    if articles_processed % 50 == 0:
                        logger.debug(
                            f"[Cache] Article {articles_processed}: "
                            f"{len(segs)} segments, total={len(self._sentence_cache)}"
                        )
                
            except StopIteration:
                logger.info(
                    f"[MultiSourceTextDataset] Source épuisée après {articles_processed} articles"
                )
                self.current_source_idx += 1
                self._initialized = False
                self._initialize_stream()
                break
                
            except Exception as e:
                logger.warning(f"[Cache] Erreur article {articles_processed}: {e}")
                continue
        
        if articles_processed >= max_articles_per_fill and len(self._sentence_cache) < self.cache_size:
            logger.warning(
                f"[Cache] Sous-rempli: {len(self._sentence_cache)}/{self.cache_size}"
            )

    def _get_text(self) -> str:
        """Récupère un texte du cache (thread-safe)."""
        with self._cache_lock:
            if len(self._sentence_cache) < self.cache_size // 2:
                self._fill_cache()
            
            if not self._sentence_cache:
                return "Texte par défaut - erreur de chargement"
            
            # Pop aléatoire pour diversité
            idx = random.randint(0, len(self._sentence_cache) - 1)
            return self._sentence_cache.pop(idx)

    def __iter__(self):
        """Générateur infini d'échantillons."""
        worker_info = torch.utils.data.get_worker_info()
        
        # Seed unique par worker
        if worker_info is not None:
            worker_seed = self.seed + worker_info.id
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)
        else:
            random.seed(self.seed)
            torch.manual_seed(self.seed)
        
        # Génération infinie
        while True:
            text = self._get_text()

            # Génère l'image (niveaux de gris)
            img = _draw_text_to_image(
                text=text,
                img_size=self.img_size,
                base_font_size=self.base_font_size,
                font_candidates=self.font_candidates,
                train=self.train,
            )
            img = self.post_color_aug(img)

            # Redimensionne/pad
            pixel_values, pixel_mask = pil_to_tensor_resize_pad(
                img, 
                target_size=self.img_size, 
                pad_value=0, 
                normalize=False, 
                return_mask=True
            )

            # Tokenisation déplacée dans le collator
            # labels = self.processor.tokenizer(
            #     text, 
            #     padding="max_length", 
            #     max_length=self.max_target_length, 
            #     truncation=True
            # ).input_ids

            yield {
                "pixel_values": pixel_values,  # (1, H, W) - niveaux de gris
                "pixel_mask": pixel_mask,       # (H, W)
                "text_label": text,             # Texte brut pour le collator
                # "labels": torch.tensor(labels, dtype=torch.long),
            }


# Alias pour compatibilité
InfiniteTextImageDataset = MultiSourceTextDataset


class WikiTextImageDataset(torch.utils.data.Dataset):
    """
    Version Map-style avec longueur fixe.
    Optimisé pour la validation avec pré-chargement limité.
    """
    
    def __init__(
        self, 
        processor,
        split: str = "train",
        max_samples: int = 1000,  # Réduit par défaut pour économiser RAM
        img_size: Union[int, Tuple[int, int]] = (32, 512),
        lang: Optional[str] = None,
        languages: Optional[List[str]] = None,
        max_target_length: int = 128,
        min_chars: int = 10,
        max_chars: int = 100,
        max_overlap: float = 0.25,
        font_candidates: Optional[List[str]] = None,
        base_font_size: int = 28,
        train: bool = True,
        seed: int = 42,
        test_size: float = 0.02,
        cache_size: int = 50000,
        article_rotation_interval: int = 1000,
        enable_all_wikipedia: bool = False,
        enable_extended_sources: bool = False,
        sources: Optional[List[Tuple[str, Optional[str], str]]] = None,
    ):
        super().__init__()
        
        self.processor = processor
        self.max_target_length = max_chars
        self.img_size = img_size
        self.font_candidates = font_candidates
        self.base_font_size = base_font_size
        self.train = train
        self.min_chars = min_chars
        self.max_chars = max_chars
        self.max_overlap = max_overlap
        self.seed = seed
        self.split = split
        self._max_samples = max_samples
        
        # Configuration des sources
        if sources is None:
            sources = []
            
            if lang is not None and languages is None:
                languages = [lang]
            
            if languages is not None:
                for lang_code in languages:
                    lang_name = lang_code.split(".")[-1].upper()
                    sources.append(("wikipedia", lang_code, f"Wikipedia-{lang_name}"))
            
            if not sources and not enable_all_wikipedia and not enable_extended_sources:
                sources = [
                    ("wikipedia", "20220301.fr", "Français"),
                    ("wikipedia", "20220301.en", "Anglais"),
                ]
            
            if enable_all_wikipedia:
                sources.extend(TextSourceConfig.WIKIPEDIA_LATIN_SOURCES)
            
            if enable_extended_sources:
                sources.extend(TextSourceConfig.OTHER_LATIN_SOURCES)
        
        self.sources = sources
        
        # Pré-chargement limité pour validation (max 5000 échantillons)
        if not train or max_samples <= 5000:
            logger.info(f"[WikiTextImageDataset] Pré-chargement de {max_samples} échantillons...")
            self._preload_samples(max_samples, test_size)
        else:
            # Mode lazy pour entraînement
            self._sentence_cache = []
            self._cache_lock = Lock()
            self._current_iterator = None
            self._initialized = False
            self._failed_sources = set()
            self.current_source_idx = 0
            self.cache_size = cache_size
            self.test_size = test_size
            self.article_rotation_interval = article_rotation_interval
            self.articles_from_current_source = 0
        
        # Augmentations
        self.post_color_aug = transforms.Compose([]) if not train else transforms.Compose([
            transforms.RandomApply(
                torch.nn.ModuleList([
                    transforms.ColorJitter(brightness=0.1, contrast=0.1),
                ]),
                p=0.3,
            )
        ])
    
    def _preload_samples(self, num_samples: int, test_size: float):
        """Pré-charge un nombre limité d'échantillons."""
        self._samples = []
        random.seed(self.seed)
        
        temp_iterator = self._create_temp_iterator(test_size)
        
        logger.info(f"[WikiTextImageDataset] Extraction de {num_samples} textes...")
        extracted = 0
        articles_read = 0
        max_articles = min(num_samples * 3, 10000)  # Limite raisonnable
        
        while len(self._samples) < num_samples and articles_read < max_articles:
            try:
                ex = next(temp_iterator)
                articles_read += 1
                
                txt = None
                for key in ["text", "content", "sentence", "paragraph"]:
                    if key in ex:
                        txt = ex[key]
                        break
                
                if not txt or len(txt) < self.min_chars:
                    continue
                
                segs = _clean_and_split_sentences(
                    txt, 
                    self.min_chars, 
                    self.max_chars,
                    self.max_overlap
                )
                self._samples.extend(segs)
                
                if articles_read % 100 == 0:
                    logger.info(f"  Articles: {articles_read}, Segments: {len(self._samples)}")
                
            except StopIteration:
                logger.info(f"[WikiTextImageDataset] Source épuisée")
                break
            except Exception as e:
                logger.warning(f"Erreur lecture: {e}")
                continue
        
        self._samples = self._samples[:num_samples]
        logger.info(f"[WikiTextImageDataset] ✓ {len(self._samples)} échantillons\n")
    
    def _create_temp_iterator(self, test_size: float):
        """Crée un itérateur temporaire pour le pré-chargement."""
        dataset_name, config_name, _ = self.sources[0]
        
        try:
            if config_name is not None:
                ds = load_dataset(
                    dataset_name, 
                    config_name, 
                    split="train", 
                    streaming=True, 
                    trust_remote_code=True
                )
            else:
                ds = load_dataset(
                    dataset_name, 
                    split="train", 
                    streaming=True, 
                    trust_remote_code=True
                )
            
            ds = ds.shuffle(seed=self.seed, buffer_size=10000)
            
            if self.split == "test":
                n_test = int(test_size * 100000)
                ds = ds.take(n_test)
            else:
                n_test = int(test_size * 100000)
                ds = ds.skip(n_test)
            
            return iter(ds)
        except Exception as e:
            logger.error(f"[WikiTextImageDataset] Erreur de chargement: {e}")
            return iter([])
    
    # === Méthodes pour le mode lazy (entraînement) ===
    
    def _get_next_source(self):
        """Obtient la prochaine source valide."""
        attempts = 0
        max_attempts = len(self.sources)
        
        while attempts < max_attempts:
            source_idx = self.current_source_idx % len(self.sources)
            if source_idx not in self._failed_sources:
                return self.sources[source_idx]
            self.current_source_idx += 1
            attempts += 1
        
        self._failed_sources.clear()
        self.current_source_idx = 0
        return self.sources[0]
    
    def _initialize_stream(self):
        """Initialise le stream pour la source courante."""
        if self._initialized:
            return
        
        dataset_name, config_name, display_name = self._get_next_source()
        
        try:
            if config_name is not None:
                ds = load_dataset(
                    dataset_name, 
                    config_name, 
                    split="train", 
                    streaming=True, 
                    trust_remote_code=True
                )
            else:
                ds = load_dataset(
                    dataset_name, 
                    split="train", 
                    streaming=True, 
                    trust_remote_code=True
                )
            
            ds = ds.shuffle(seed=self.seed + self.current_source_idx, buffer_size=10000)
            
            if self.split == "test":
                n_test = int(self.test_size * 100000)
                ds = ds.take(n_test)
            else:
                n_test = int(self.test_size * 100000)
                ds = ds.skip(n_test)
            
            self._current_iterator = iter(ds)
            self._initialized = True
            self.articles_from_current_source = 0
            
        except Exception as e:
            logger.error(f"Erreur {display_name}: {e}")
            self._failed_sources.add(self.current_source_idx % len(self.sources))
            self.current_source_idx += 1
            self._initialized = False
            self._initialize_stream()
    
    def _rotate_source_if_needed(self):
        """Change de source si l'intervalle est atteint."""
        if self.articles_from_current_source >= self.article_rotation_interval:
            self.current_source_idx += 1
            self._initialized = False
            self._current_iterator = None
            self.articles_from_current_source = 0
    
    def _fill_cache(self):
        """Remplit le cache avec de nouvelles phrases."""
        if not self._initialized:
            self._initialize_stream()
        
        self._rotate_source_if_needed()
        
        articles_processed = 0
        max_articles_per_fill = 100
        
        while len(self._sentence_cache) < self.cache_size and articles_processed < max_articles_per_fill:
            try:
                ex = next(self._current_iterator)
                articles_processed += 1
                self.articles_from_current_source += 1
                
                txt = None
                for key in ["text", "content", "sentence", "paragraph"]:
                    if key in ex:
                        txt = ex[key]
                        break
                
                if not txt or len(txt) < self.min_chars:
                    continue
                
                segs = _clean_and_split_sentences(
                    txt, 
                    self.min_chars, 
                    self.max_chars,
                    self.max_overlap
                )
                
                if len(segs) > 0:
                    self._sentence_cache.extend(segs)
                
            except StopIteration:
                self.current_source_idx += 1
                self._initialized = False
                self._initialize_stream()
                break
            except Exception as e:
                logger.warning(f"Erreur lecture: {e}")
                continue
    
    def _get_text(self) -> str:
        """Récupère un texte (lazy ou pré-chargé)."""
        if hasattr(self, '_samples'):
            # Mode pré-chargé (validation)
            idx = random.randint(0, len(self._samples) - 1)
            return self._samples[idx]
        else:
            # Mode lazy (entraînement) - thread-safe
            with self._cache_lock:
                if len(self._sentence_cache) < self.cache_size // 2:
                    self._fill_cache()
                
                if not self._sentence_cache:
                    return "Texte par défaut"
                
                idx = random.randint(0, len(self._sentence_cache) - 1)
                return self._sentence_cache.pop(idx)
    
    def __len__(self) -> int:
        return self._max_samples
    
    def __getitem__(self, idx: int):
        """Génère un échantillon de manière reproductible."""
        # Générateur local pour ne pas affecter le RNG global
        rng = random.Random(self.seed + idx)
        
        if hasattr(self, '_samples'):
            # Mode validation: utilise directement l'index
            text = self._samples[idx % len(self._samples)]
        else:
            # Mode entraînement: génération avec RNG local
            # Note: _get_text() utilise random.randint(), on doit synchroniser
            old_state = random.getstate()
            random.setstate(rng.getstate())
            text = self._get_text()
            random.setstate(old_state)
        
        img = _draw_text_to_image(
            text=text,
            img_size=self.img_size,
            base_font_size=self.base_font_size,
            font_candidates=self.font_candidates,
            train=self.train,
        )
        img = self.post_color_aug(img)

        pixel_values, pixel_mask = pil_to_tensor_resize_pad(
            img, 
            target_size=self.img_size, 
            pad_value=0, 
            normalize=False, 
            return_mask=True
        )

        # Tokenisation déplacée dans le collator
        # labels = self.processor.tokenizer(
        #     text, 
        #     padding="max_length", 
        #     max_length=self.max_target_length, 
        #     truncation=True
        # ).input_ids

        return {
            "pixel_values": pixel_values,  # (1, H, W) - niveaux de gris
            "pixel_mask": pixel_mask,       # (H, W)
            "text_label": text,             # Texte brut pour le collator
            # "labels": torch.tensor(labels, dtype=torch.long),
        }