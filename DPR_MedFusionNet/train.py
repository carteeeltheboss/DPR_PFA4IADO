"""
train.py
========
Script d'entraînement pour MedFusionNet.

Fonctionnalités :
    - Gestion du déséquilibre de classes (Focal Loss + WeightedRandomSampler)
    - Scheduling adaptatif du learning rate (CosineAnnealingWarmRestarts)
    - Warmup linéaire des 3 premières époques
    - Sauvegarde des meilleurs checkpoints (AUC-ROC)
    - Logging structuré (CSV + console)
    - Support mixed-precision (torch.cuda.amp)
    - Early stopping avec patience configurable
    - Calcul de la CAM pour ℒ_loc (si bbox disponibles)

Usage :
    python train.py --data_dir ./data --epochs 50 --batch_size 16 --lr 1e-4
    python train.py --data_dir ./data --resume checkpoints/best.pth
"""

import os
import sys
import ssl
import csv

# Bypass SSL certificate verification for downloading pretrained weights (macOS fix)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
import time
import argparse
import logging
from pathlib import Path
from typing import Optional, Dict
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

# ── Modules locaux ─────────────────────────────────────────────────────────────
from dataset import build_dataloaders
from model   import MedFusionNet
from losses  import MedFusionLoss


# ─── Configuration de logging ──────────────────────────────────────────────────
def setup_logging(log_dir: str) -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file  = Path(log_dir) / f"train_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("MedFusionNet")


# ─── Arguments CLI ─────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entraînement de MedFusionNet pour la détection de pneumonie",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Données
    parser.add_argument("--data_dir",  type=str, default="./data",
                        help="Répertoire racine du dataset (contient train/, val/, test/)")
    parser.add_argument("--bbox_file", type=str, default=None,
                        help="Fichier CSV avec annotations bbox (x1,y1,x2,y2)")
    parser.add_argument("--img_size",  type=int, default=384,
                        help="Taille des images (default: 384)")
    parser.add_argument("--no_clahe",  action="store_true",
                        help="Désactiver CLAHE")

    # Modèle
    parser.add_argument("--pretrained",       action="store_true", default=True,
                        help="Utiliser les poids ImageNet")
    parser.add_argument("--freeze_cnn",       type=int, default=2,
                        help="Nombre de dense blocks à geler (0-4)")
    parser.add_argument("--dropout",          type=float, default=0.3,
                        help="Taux de Dropout")

    # Entraînement
    parser.add_argument("--epochs",           type=int,   default=50)
    parser.add_argument("--batch_size",       type=int,   default=4)
    parser.add_argument("--lr",               type=float, default=1e-4,
                        help="Learning rate initial")
    parser.add_argument("--lr_min",           type=float, default=1e-6,
                        help="Learning rate minimal (CosineAnnealing)")
    parser.add_argument("--warmup_epochs",    type=int,   default=3,
                        help="Époques de warmup linéaire")
    parser.add_argument("--weight_decay",     type=float, default=1e-4)
    parser.add_argument("--grad_clip",        type=float, default=1.0,
                        help="Norme max du gradient (clipping)")
    parser.add_argument("--patience",         type=int,   default=10,
                        help="Patience pour l'early stopping")
    parser.add_argument("--num_workers",      type=int,   default=0)
    parser.add_argument("--amp",              action="store_true", default=False,
                        help="Mixed precision training (AMP)")

    # Pertes
    parser.add_argument("--lambda_loc",       type=float, default=0.5)
    parser.add_argument("--lambda_cons",      type=float, default=0.3)
    parser.add_argument("--lambda_cal",       type=float, default=0.1)
    parser.add_argument("--focal_alpha",      type=float, default=0.25)
    parser.add_argument("--focal_gamma",      type=float, default=2.0)

    # I/O
    parser.add_argument("--output_dir",  type=str, default="./outputs")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir",     type=str, default="./logs")
    parser.add_argument("--resume",      type=str, default=None,
                        help="Chemin vers un checkpoint à reprendre")
    parser.add_argument("--seed",        type=int, default=42)

    return parser.parse_args()


# ─── Reproductibilité ──────────────────────────────────────────────────────────
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Warmup + CosineAnnealing scheduler custom ─────────────────────────────────
def build_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    lr_min: float,
) -> optim.lr_scheduler.LambdaLR:
    """
    Combine warmup linéaire et CosineAnnealingLR.

    Phases :
        Époques 0 → warmup_epochs : LR augmente linéairement de 0 à lr
        Époques warmup_epochs → total : CosineAnnealing vers lr_min
    """
    def lr_lambda(epoch: int) -> float:
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(warmup_epochs)   # Warmup
        # Cosine decay
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
        # Facteur ramenant lr de 1 → lr_min/lr
        return lr_min / optimizer.param_groups[0]["lr"] + \
               cosine * (1.0 - lr_min / optimizer.param_groups[0]["lr"])

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ─── Métriques ─────────────────────────────────────────────────────────────────
def compute_metrics(
    all_probs: np.ndarray,
    all_preds: np.ndarray,
    all_labels: np.ndarray,
) -> dict:
    """Calcule AUC-ROC, F1, Accuracy."""
    metrics = {
        "acc"  : accuracy_score(all_labels, all_preds),
        "f1"   : f1_score(all_labels, all_preds, zero_division=0),
        "auc"  : 0.0,
    }
    if len(np.unique(all_labels)) > 1:
        metrics["auc"] = roc_auc_score(all_labels, all_probs)
    return metrics


# ─── Epoch d'entraînement ──────────────────────────────────────────────────────
def train_one_epoch(
    model:      MedFusionNet,
    loader:     torch.utils.data.DataLoader,
    optimizer:  optim.Optimizer,
    criterion:  MedFusionLoss,
    scaler:     GradScaler,
    device:     torch.device,
    grad_clip:  float,
    logger:     logging.Logger,
    use_amp:    bool = True,
) -> dict:
    """
    Effectue une époque d'entraînement complète.

    Returns:
        Dictionnaire de statistiques moyennées : loss total, composantes, métriques
    """
    model.train()

    # Accumulateurs
    total_loss  = 0.0
    loss_parts  = {"cls": 0.0, "loc": 0.0, "cons": 0.0, "cal": 0.0}
    all_probs   = []
    all_preds   = []
    all_labels  = []
    n_batches   = len(loader)
    t0 = time.time()

    for batch_idx, batch in enumerate(loader):
        images    = batch["image"].to(device, non_blocking=True)      # (B,3,H,W)
        labels    = batch["label"].to(device, non_blocking=True)      # (B,)
        bboxes    = batch["bbox"].to(device, non_blocking=True)       # (B,4)
        bbox_mask = (bboxes.sum(dim=1) > 0).float()                   # (B,)

        optimizer.zero_grad(set_to_none=True)

        # ── Forward pass (mixed precision) ────────────────────────────────
        with autocast(enabled=use_amp):
            out = model(images)

            losses = criterion(
                logits     = out["logit"],
                targets    = labels,
                model_out  = out,
                cam        = None,
                bboxes     = bboxes,
                bbox_mask  = bbox_mask,
            )
            loss = losses["total"]

        # ── 2. Backward ──
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()

        # Nettoyage mémoire MPS (Mac)
        if device.type == "mps" and batch_idx % 10 == 0:
            torch.mps.empty_cache()

        # ── Accumulation ──────────────────────────────────────────────────
        total_loss += loss.item()
        for key in loss_parts:
            loss_parts[key] += losses[key].item()

        probs = out["prob"].squeeze(1).detach().cpu().float().numpy()
        preds = (probs >= 0.5).astype(int)
        labs  = labels.cpu().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend(preds.tolist())
        all_labels.extend(labs.tolist())

        # Log toutes les 10 itérations
        if (batch_idx + 1) % max(1, n_batches // 5) == 0:
            elapsed = time.time() - t0
            logger.info(
                f"  [Train] {batch_idx+1:4d}/{n_batches} | "
                f"Loss: {loss.item():.4f} | "
                f"Cls: {losses['cls'].item():.4f} | "
                f"Cons: {losses['cons'].item():.4f} | "
                f"Cal: {losses['cal'].item():.4f} | "
                f"Temps: {elapsed:.1f}s"
            )

    # ── Statistiques finales ───────────────────────────────────────────────
    n  = n_batches
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_probs, all_preds, all_labels)
    return {
        "loss"  : total_loss / n,
        "cls"   : loss_parts["cls"]  / n,
        "loc"   : loss_parts["loc"]  / n,
        "cons"  : loss_parts["cons"] / n,
        "cal"   : loss_parts["cal"]  / n,
        **metrics,
    }


# ─── Epoch de validation ───────────────────────────────────────────────────────
@torch.no_grad()
def validate(
    model:     MedFusionNet,
    loader:    torch.utils.data.DataLoader,
    criterion: MedFusionLoss,
    device:    torch.device,
    use_amp:   bool = True,
) -> dict:
    """
    Évalue le modèle sur le jeu de validation.

    Returns:
        Dictionnaire de métriques
    """
    model.eval()

    total_loss = 0.0
    loss_parts = {"cls": 0.0, "loc": 0.0, "cons": 0.0, "cal": 0.0}
    all_probs  = []
    all_preds  = []
    all_labels = []

    for batch in loader:
        images    = batch["image"].to(device, non_blocking=True)
        labels    = batch["label"].to(device, non_blocking=True)
        bboxes    = batch["bbox"].to(device, non_blocking=True)
        bbox_mask = (bboxes.sum(dim=1) > 0).float()

        with autocast(enabled=use_amp):
            out    = model(images)
            losses = criterion(out["logit"], labels, out, None, bboxes, bbox_mask)

        total_loss += losses["total"].item()
        for key in loss_parts:
            loss_parts[key] += losses[key].item()

        probs = out["prob"].squeeze(1).cpu().float().numpy()
        all_probs.extend(probs.tolist())
        all_preds.extend((probs >= 0.5).astype(int).tolist())
        all_labels.extend(labels.cpu().numpy().tolist())

    n = len(loader)
    all_probs  = np.array(all_probs)
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    metrics = compute_metrics(all_probs, all_preds, all_labels)
    return {
        "loss"  : total_loss / n,
        "cls"   : loss_parts["cls"]  / n,
        "loc"   : loss_parts["loc"]  / n,
        "cons"  : loss_parts["cons"] / n,
        "cal"   : loss_parts["cal"]  / n,
        **metrics,
    }


# ─── Sauvegarde de checkpoint ──────────────────────────────────────────────────
def save_checkpoint(
    model:     nn.Module,
    optimizer: optim.Optimizer,
    scheduler,
    epoch:     int,
    metrics:   dict,
    args:      argparse.Namespace,
    path:      str,
):
    """Sauvegarde le modèle, l'optimiseur, et les métadonnées."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch"     : epoch,
        "model_state": model.state_dict(),
        "optim_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "metrics"   : metrics,
        "args"      : vars(args),
    }, path)


# ─── Loop d'entraînement principal ─────────────────────────────────────────────
def train(args: argparse.Namespace):
    # ── Setup ────────────────────────────────────────────────────────────────
    set_seed(args.seed)
    logger = setup_logging(args.log_dir)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Device : {device} ({torch.cuda.get_device_name(0)})")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info(f"Device : {device} (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        logger.info(f"Device : {device}")

    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # ── DataLoaders ───────────────────────────────────────────────────────────
    logger.info("Chargement du dataset...")
    loaders = build_dataloaders(
        data_dir             = args.data_dir,
        batch_size           = args.batch_size,
        img_size             = args.img_size,
        num_workers          = args.num_workers,
        use_clahe            = not args.no_clahe,
        use_weighted_sampler = True,      # Correction du déséquilibre de classes
        bbox_file            = args.bbox_file,
        pin_memory           = device.type == "cuda",
    )

    train_loader = loaders.get("train")
    val_loader   = loaders.get("val")

    if train_loader is None:
        raise RuntimeError("DataLoader d'entraînement manquant !")

    # ── Modèle ────────────────────────────────────────────────────────────────
    logger.info("Initialisation du modèle MedFusionNet...")
    model = MedFusionNet(
        num_classes       = 1,
        dropout_rate      = args.dropout,
        pretrained        = args.pretrained,
        freeze_cnn_blocks = args.freeze_cnn,
    ).to(device)

    params = model.count_parameters()
    logger.info(f"Paramètres totaux entraînables : {params['total']:,}")

    # DataParallel si multi-GPU
    if torch.cuda.device_count() > 1:
        logger.info(f"Utilisation de {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    # ── Pertes ────────────────────────────────────────────────────────────────
    criterion = MedFusionLoss(
        lambda_loc  = args.lambda_loc,
        lambda_cons = args.lambda_cons,
        lambda_cal  = args.lambda_cal,
        focal_alpha = args.focal_alpha,
        focal_gamma = args.focal_gamma,
    )

    # ── Optimiseur ────────────────────────────────────────────────────────────
    # Learning rates différenciés : backbone pré-entraîné → LR/10
    backbone_params = []
    head_params     = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "local_branch.features" in name or "swin" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    optimizer = optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr / 10},  # Fine-tuning prudent
            {"params": head_params,     "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    scheduler = build_scheduler(
        optimizer     = optimizer,
        warmup_epochs = args.warmup_epochs,
        total_epochs  = args.epochs,
        lr_min        = args.lr_min,
    )

    # ── Mixed Precision Scaler ─────────────────────────────────────────────────
    scaler = GradScaler(enabled=args.amp and device.type == "cuda")

    # ── Reprise d'entraînement ────────────────────────────────────────────────
    start_epoch   = 0
    best_auc      = 0.0
    patience_cnt  = 0

    if args.resume and Path(args.resume).exists():
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optim_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        start_epoch  = ckpt["epoch"] + 1
        best_auc     = ckpt["metrics"].get("auc", 0.0)
        logger.info(f"Reprise depuis l'époque {start_epoch} | AUC: {best_auc:.4f}")

    # ── CSV de logging ─────────────────────────────────────────────────────────
    csv_path = Path(args.log_dir) / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", "lr",
        "train_loss", "train_cls", "train_cons", "train_cal",
        "train_acc",  "train_f1",  "train_auc",
        "val_loss",   "val_cls",   "val_cons",   "val_cal",
        "val_acc",    "val_f1",    "val_auc",
    ])

    # ═══════════════════════════════════════════════════════════════════════════
    # BOUCLE D'ENTRAÎNEMENT PRINCIPALE
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info("=" * 65)
    logger.info(f"  Démarrage de l'entraînement | {args.epochs} époques")
    logger.info("=" * 65)

    for epoch in range(start_epoch, args.epochs):
        current_lr = optimizer.param_groups[-1]["lr"]   # LR du groupe principal
        logger.info(f"\n{'─'*60}")
        logger.info(f"  Époque {epoch+1:3d}/{args.epochs} | LR = {current_lr:.2e}")
        logger.info(f"{'─'*60}")

        # ── Entraînement ──────────────────────────────────────────────────────
        train_m = train_one_epoch(
            model, train_loader, optimizer, criterion,
            scaler, device, args.grad_clip, logger, args.amp
        )

        # ── Validation ────────────────────────────────────────────────────────
        val_m = {"loss": 0., "cls": 0., "loc": 0., "cons": 0., "cal": 0.,
                 "acc": 0., "f1": 0., "auc": 0.}
        if val_loader is not None:
            val_m = validate(model, val_loader, criterion, device, args.amp)

        # ── Scheduling ────────────────────────────────────────────────────────
        scheduler.step()

        # ── Logging ───────────────────────────────────────────────────────────
        logger.info(
            f"\n  TRAIN | Loss: {train_m['loss']:.4f} | "
            f"AUC: {train_m['auc']:.4f} | F1: {train_m['f1']:.4f} | "
            f"Acc: {train_m['acc']:.4f}"
        )
        logger.info(
            f"  VAL   | Loss: {val_m['loss']:.4f} | "
            f"AUC: {val_m['auc']:.4f} | F1: {val_m['f1']:.4f} | "
            f"Acc: {val_m['acc']:.4f}"
        )

        csv_writer.writerow([
            epoch + 1,         current_lr,
            train_m["loss"],   train_m["cls"],  train_m["cons"],  train_m["cal"],
            train_m["acc"],    train_m["f1"],   train_m["auc"],
            val_m["loss"],     val_m["cls"],    val_m["cons"],    val_m["cal"],
            val_m["acc"],      val_m["f1"],     val_m["auc"],
        ])
        csv_file.flush()

        # ── Sauvegarde du meilleur modèle (basé sur AUC-ROC) ─────────────────
        if val_m["auc"] > best_auc:
            best_auc     = val_m["auc"]
            patience_cnt = 0
            best_path    = Path(args.checkpoint_dir) / "best.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, val_m, args, str(best_path))
            logger.info(f"  ✅ Nouveau meilleur modèle → AUC = {best_auc:.4f} [{best_path}]")
        else:
            patience_cnt += 1
            logger.info(f"  ⚠️  Patience : {patience_cnt}/{args.patience}")

        # Checkpoint périodique (toutes les 10 époques)
        if (epoch + 1) % 10 == 0:
            periodic_path = Path(args.checkpoint_dir) / f"epoch_{epoch+1:03d}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch, val_m, args, str(periodic_path))

        # ── Early Stopping ────────────────────────────────────────────────────
        if patience_cnt >= args.patience:
            logger.info(f"\n  🛑 Early stopping à l'époque {epoch+1} "
                        f"(patience={args.patience})")
            break

    # ── Fin d'entraînement ────────────────────────────────────────────────────
    csv_file.close()
    logger.info("\n" + "=" * 65)
    logger.info(f"  Entraînement terminé ! Meilleur AUC : {best_auc:.4f}")
    logger.info(f"  Meilleur modèle : {args.checkpoint_dir}/best.pth")
    logger.info(f"  Métriques CSV   : {csv_path}")
    logger.info("=" * 65)


# ─── Point d'entrée ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()
    train(args)
