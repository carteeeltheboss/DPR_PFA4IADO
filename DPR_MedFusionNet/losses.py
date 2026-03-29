"""
losses.py
=========
Fonctions de perte pour MedFusionNet.

Perte totale :
    ℒ = ℒ_cls + λ₁·ℒ_loc + λ₂·ℒ_cons + λ₃·ℒ_cal

    ℒ_cls  : Classification — Focal Loss (robuste au déséquilibre de classes)
    ℒ_loc  : Localisation   — Cohérence entre Grad-CAM et boîtes englobantes
    ℒ_cons : Consistance    — Cohérence entre les prédictions des deux branches
    ℒ_cal  : Calibration    — Expected Calibration Error différentiable

Références :
    Focal Loss : Lin et al., 2017 — "Focal Loss for Dense Object Detection"
    ECE Loss   : Guo et al., 2017 — "On Calibration of Modern Neural Networks"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 1. FOCAL LOSS  (ℒ_cls)
# ═══════════════════════════════════════════════════════════════════════════════
class FocalLoss(nn.Module):
    """
    Focal Loss pour la classification binaire déséquilibrée.

    FL(p_t) = -α_t · (1 - p_t)^γ · log(p_t)

    Args:
        alpha : Facteur de pondération de la classe positive (défaut: 0.25)
        gamma : Facteur de focus sur les exemples difficiles (défaut: 2.0)
        reduction: 'mean' | 'sum' | 'none'

    Notes :
        - γ = 0 → Focal Loss = Cross-Entropy classique
        - γ = 2 → Configuration recommandée (Lin et al., 2017)
        - α = 0.75 si PN >> PP (plus de négatifs que de positifs)
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean",
    ):
        super().__init__()
        assert reduction in ("mean", "sum", "none"), \
            f"reduction doit être 'mean', 'sum' ou 'none', reçu {reduction}"
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,   # (B, 1) — logits bruts
        targets: torch.Tensor,  # (B,)   — labels binaires {0, 1}
    ) -> torch.Tensor:
        """
        Args:
            logits : Logits non-sigmoïdés (B, 1) ou (B,)
            targets: Étiquettes binaires  (B,)

        Returns:
            Scalaire de perte
        """
        logits = logits.squeeze(1)                          # (B,)
        targets = targets.float()                          # (B,)

        # Probabilités stables numériquement via BCE with logits
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )                                                  # (B,)

        # p_t : probabilité de la classe correcte
        probs  = torch.sigmoid(logits)
        p_t    = probs * targets + (1 - probs) * (1 - targets)   # (B,)

        # Facteur alpha : α si positif, (1-α) si négatif
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)  # (B,)

        # Focal weight : (1 - p_t)^γ
        focal_weight = (1.0 - p_t) ** self.gamma          # (B,)

        # Focal Loss
        focal_loss = alpha_t * focal_weight * bce_loss     # (B,)

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ═══════════════════════════════════════════════════════════════════════════════
# 2. PERTE DE LOCALISATION  (ℒ_loc)
# ═══════════════════════════════════════════════════════════════════════════════
class LocalizationLoss(nn.Module):
    """
    Perte de cohérence entre la carte de saillance (Grad-CAM) et les
    boîtes englobantes annotées (supervision faible).

    Formule :
        ℒ_loc = BCE(CAM_norm, M_bbox)

    où M_bbox est un masque binaire créé à partir de la bbox annotée,
    et CAM_norm est la heatmap Grad-CAM normalisée dans [0, 1].

    Note : Utilisée uniquement pour les images ayant une bbox annotée.
    """

    def __init__(self, cam_size: int = 12):
        """
        Args:
            cam_size: Résolution spatiale des CAM (défaut: 12 pour entrée 384×384)
        """
        super().__init__()
        self.cam_size = cam_size

    def forward(
        self,
        cam: torch.Tensor,       # (B, 1, h, w) — heatmap Grad-CAM
        bboxes: torch.Tensor,    # (B, 4)        — [x1, y1, x2, y2] normalisé [0,1]
        bbox_mask: torch.Tensor, # (B,)          — 1 si bbox disponible, 0 sinon
    ) -> torch.Tensor:
        """
        Args:
            cam      : Grad-CAM heatmap (B, 1, h, w), valeurs dans [0, 1]
            bboxes   : Coordonnées bbox normalisées (B, 4)
            bbox_mask: Masque indiquant quelles images ont des bbox (B,)

        Returns:
            Scalaire de perte (0. si aucune bbox disponible)
        """
        B = cam.shape[0]
        h = w = self.cam_size

        if bbox_mask.sum() == 0:
            return cam.new_tensor(0.0)

        # Créer les masques binaires à partir des bbox
        target_mask = self._bbox_to_mask(bboxes, h, w, B)   # (B, 1, h, w)

        # Normaliser la CAM dans [0, 1]
        cam_min = cam.flatten(1).min(dim=1)[0].view(B, 1, 1, 1)
        cam_max = cam.flatten(1).max(dim=1)[0].view(B, 1, 1, 1)
        cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        # Appliquer uniquement aux images avec bbox
        mask = bbox_mask.float().view(B, 1, 1, 1)
        loss = F.binary_cross_entropy(
            cam_norm * mask,
            target_mask * mask,
            reduction="sum",
        ) / (bbox_mask.sum() * h * w + 1e-8)

        return loss

    @staticmethod
    def _bbox_to_mask(
        bboxes: torch.Tensor,   # (B, 4) normalisé [0, 1]
        h: int,
        w: int,
        B: int,
    ) -> torch.Tensor:
        """Convertit les bbox normalisées en masques binaires (B, 1, h, w)."""
        mask = torch.zeros(B, 1, h, w, device=bboxes.device)
        for i in range(B):
            x1, y1, x2, y2 = bboxes[i]
            c1 = int(x1 * w)
            r1 = int(y1 * h)
            c2 = int(x2 * w)
            r2 = int(y2 * h)
            mask[i, 0, r1:r2, c1:c2] = 1.0
        return mask


# ═══════════════════════════════════════════════════════════════════════════════
# 3. PERTE DE CONSISTANCE  (ℒ_cons)
# ═══════════════════════════════════════════════════════════════════════════════
class ConsistencyLoss(nn.Module):
    """
    Perte de consistance entre les prédictions des deux branches (CNN et Swin).

    Encourage les deux branches à produire des prédictions cohérentes,
    agissant comme une régularisation pour éviter la sur-spécialisation.

    Formule :
        ℒ_cons = KL(p_c || p_mean) + KL(p_s || p_mean)

    où p_mean = (p_c + p_s) / 2 est la prédiction moyenne.

    Alternative MSE (plus stable) :
        ℒ_cons = ||p_c - p_s||²
    """

    def __init__(self, mode: str = "kl"):
        """
        Args:
            mode: 'kl' (KL divergence) | 'mse' (Mean Squared Error)
        """
        super().__init__()
        assert mode in ("kl", "mse")
        self.mode = mode

        # Tête de classification auxiliaire pour la branche CNN
        # (partagée avec le modèle principal via injection)
        self.fc_local  = None
        self.fc_global = None

    def set_aux_classifiers(
        self,
        fc_local: nn.Linear,
        fc_global: nn.Linear,
    ):
        """Injecte les classifieurs auxiliaires des branches."""
        self.fc_local  = fc_local
        self.fc_global = fc_global

    def forward(
        self,
        gap_c: torch.Tensor,   # (B, D) — features poolées CNN
        gap_s: torch.Tensor,   # (B, D) — features poolées Swin
    ) -> torch.Tensor:
        """
        Args:
            gap_c: GAP des features CNN  (B, D)
            gap_s: GAP des features Swin (B, D)

        Returns:
            Scalaire de perte de consistance
        """
        # Prédictions des branches individuelles via leur classifieur auxiliaire
        if self.fc_local is None or self.fc_global is None:
            # Fallback : MSE entre features normalisées
            gap_c_norm = F.normalize(gap_c, dim=1)
            gap_s_norm = F.normalize(gap_s, dim=1)
            return F.mse_loss(gap_c_norm, gap_s_norm)

        p_c = torch.sigmoid(self.fc_local(gap_c)).squeeze(1)    # (B,)
        p_s = torch.sigmoid(self.fc_global(gap_s)).squeeze(1)   # (B,)

        if self.mode == "mse":
            return F.mse_loss(p_c, p_s)

        # KL divergence symétrique (Jensen-Shannon)
        p_mean = 0.5 * (p_c + p_s).clamp(1e-6, 1 - 1e-6)
        p_c_c  = p_c.clamp(1e-6, 1 - 1e-6)
        p_s_c  = p_s.clamp(1e-6, 1 - 1e-6)

        kl_c = p_c_c * (p_c_c / p_mean).log() + \
               (1 - p_c_c) * ((1 - p_c_c) / (1 - p_mean)).log()
        kl_s = p_s_c * (p_s_c / p_mean).log() + \
               (1 - p_s_c) * ((1 - p_s_c) / (1 - p_mean)).log()

        return 0.5 * (kl_c.mean() + kl_s.mean())


# ═══════════════════════════════════════════════════════════════════════════════
# 4. PERTE DE CALIBRATION  (ℒ_cal)
# ═══════════════════════════════════════════════════════════════════════════════
class CalibrationLoss(nn.Module):
    """
    Perte de calibration différentiable basée sur l'Expected Calibration Error (ECE).

    Objectif : Aligner les confidences prédites p avec les précisions observées,
    pour que p = 0.8 signifie vraiment 80% de chance d'être correct.

    Formule (soft ECE) :
        ℒ_cal = Σ_m  (n_m / N) · |acc_m - conf_m|

    où les bins are softement attribués via une fonction gaussienne.

    Référence : Guo et al., ICML 2017
    """

    def __init__(self, n_bins: int = 10):
        """
        Args:
            n_bins: Nombre de bins de calibration (défaut: 10)
        """
        super().__init__()
        self.n_bins       = n_bins
        self.bin_centers  = torch.linspace(0.05, 0.95, n_bins)   # Centres fixes
        self.bin_width    = 1.0 / n_bins

    def forward(
        self,
        probs: torch.Tensor,    # (B,) ou (B, 1) — probabilités sigmoid
        targets: torch.Tensor,  # (B,)             — labels binaires {0, 1}
    ) -> torch.Tensor:
        """
        Args:
            probs  : Probabilités prédites (B,)
            targets: Labels vrais (B,)

        Returns:
            Scalaire de perte ECE différentiable
        """
        probs   = probs.squeeze(1).float()
        targets = targets.float()
        B       = probs.shape[0]

        centers = self.bin_centers.to(probs.device)   # (n_bins,)

        # Attribution soft aux bins via kernel gaussien
        # w_b(i) = exp(-(p_i - c_b)² / (2·σ²))
        sigma  = self.bin_width / 2
        # (B, n_bins) — poids d'appartenance pour chaque exemple dans chaque bin
        diffs  = (probs.unsqueeze(1) - centers.unsqueeze(0)) ** 2  # (B, n_bins)
        w      = torch.exp(-diffs / (2 * sigma ** 2))               # (B, n_bins)
        w      = w / (w.sum(dim=1, keepdim=True) + 1e-8)            # normalisation

        # Confiance et précision par bin
        # conf_b = Σ_i w_b(i) · p_i / Σ_i w_b(i)
        # acc_b  = Σ_i w_b(i) · y_i / Σ_i w_b(i)
        n_bin = w.sum(dim=0)                        # (n_bins,)
        conf  = (w * probs.unsqueeze(1)).sum(0) / (n_bin + 1e-8)    # (n_bins,)
        acc   = (w * targets.unsqueeze(1)).sum(0) / (n_bin + 1e-8)  # (n_bins,)

        # Poids de chaque bin par proportion d'exemples
        prop_bin = n_bin / (B + 1e-8)              # (n_bins,)

        # ECE différentiable
        ece = (prop_bin * (conf - acc).abs()).sum()
        return ece


# ═══════════════════════════════════════════════════════════════════════════════
# 5. PERTE TOTALE — MedFusionLoss
# ═══════════════════════════════════════════════════════════════════════════════
class MedFusionLoss(nn.Module):
    """
    Perte combinée pour MedFusionNet :

        ℒ = ℒ_cls + λ₁·ℒ_loc + λ₂·ℒ_cons + λ₃·ℒ_cal

    Args :
        lambda_loc  : Poids de ℒ_loc  (défaut: 0.5)
        lambda_cons : Poids de ℒ_cons (défaut: 0.3)
        lambda_cal  : Poids de ℒ_cal  (défaut: 0.1)
        focal_alpha : Paramètre α de la Focal Loss
        focal_gamma : Paramètre γ de la Focal Loss
        n_bins      : Nombre de bins pour ℒ_cal
    """

    def __init__(
        self,
        lambda_loc:  float = 0.5,
        lambda_cons: float = 0.3,
        lambda_cal:  float = 0.1,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        n_bins:      int   = 10,
    ):
        super().__init__()

        self.lambda_loc  = lambda_loc
        self.lambda_cons = lambda_cons
        self.lambda_cal  = lambda_cal

        # Composantes de la perte
        self.focal_loss   = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        self.loc_loss     = LocalizationLoss(cam_size=12)
        self.cons_loss    = ConsistencyLoss(mode="mse")
        self.cal_loss     = CalibrationLoss(n_bins=n_bins)

    def forward(
        self,
        logits:   torch.Tensor,                  # (B, 1)
        targets:  torch.Tensor,                  # (B,)
        model_out: dict,                         # Sortie complète du modèle
        cam:      Optional[torch.Tensor] = None, # (B, 1, h, w) Grad-CAM
        bboxes:   Optional[torch.Tensor] = None, # (B, 4)
        bbox_mask: Optional[torch.Tensor] = None,# (B,)
    ) -> dict:
        """
        Args:
            logits    : Logits bruts du modèle (B, 1)
            targets   : Labels binaires (B,)
            model_out : Dictionnaire de sorties du modèle (f_c, f_s, gap_feat...)
            cam       : Heatmap Grad-CAM (B, 1, h, w) — optionnel
            bboxes    : Boîtes englobantes normalisées (B, 4) — optionnel
            bbox_mask : Masque de disponibilité des bbox (B,) — optionnel

        Returns:
            Dictionnaire avec la perte totale et les composantes individuelles
        """
        # ── ℒ_cls : Focal Loss ──────────────────────────────────────────────
        l_cls = self.focal_loss(logits, targets)

        # ── ℒ_loc : Localisation (si CAM et bbox disponibles) ──────────────
        l_loc = logits.new_tensor(0.0)
        if cam is not None and bboxes is not None and bbox_mask is not None:
            l_loc = self.loc_loss(cam, bboxes, bbox_mask)

        # ── ℒ_cons : Consistance inter-branches ─────────────────────────────
        gap_c = model_out["f_c"].mean(dim=[2, 3])   # (B, D) — GAP de F_c
        gap_s = model_out["f_s"].mean(dim=[2, 3])   # (B, D) — GAP de F_s
        l_cons = self.cons_loss(gap_c, gap_s)

        # ── ℒ_cal : Calibration ──────────────────────────────────────────────
        probs  = model_out["prob"]
        l_cal  = self.cal_loss(probs, targets)

        # ── Perte totale ─────────────────────────────────────────────────────
        l_total = (
            l_cls
            + self.lambda_loc  * l_loc
            + self.lambda_cons * l_cons
            + self.lambda_cal  * l_cal
        )

        return {
            "total" : l_total,
            "cls"   : l_cls.detach(),
            "loc"   : l_loc.detach(),
            "cons"  : l_cons.detach(),
            "cal"   : l_cal.detach(),
        }


# ─── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  MedFusionNet — Test des fonctions de perte")
    print("=" * 60)

    B = 8
    device = torch.device("cpu")

    # Données synthétiques
    logits  = torch.randn(B, 1)
    targets = torch.randint(0, 2, (B,))
    probs   = torch.sigmoid(logits)
    f_c     = torch.randn(B, 512, 12, 12)
    f_s     = torch.randn(B, 512, 12, 12)
    cam     = torch.sigmoid(torch.randn(B, 1, 12, 12))  # CAM normalisée
    bboxes  = torch.rand(B, 4)
    bboxes[:, 2:] = bboxes[:, 2:].clamp(min=bboxes[:, :2] + 0.1)  # x2>x1, y2>y1
    bbox_mask = torch.ones(B)

    model_out = {"prob": probs, "f_c": f_c, "f_s": f_s}

    criterion = MedFusionLoss(lambda_loc=0.5, lambda_cons=0.3, lambda_cal=0.1)
    losses = criterion(logits, targets, model_out, cam, bboxes, bbox_mask)

    print(f"\n  Résultats :")
    for name, val in losses.items():
        print(f"    ℒ_{name:<6} = {val.item():.4f}")

    # Test Focal Loss seul
    fl = FocalLoss(alpha=0.25, gamma=2.0)
    print(f"\n  Focal Loss γ=2 : {fl(logits, targets).item():.4f}")
    fl_ce = FocalLoss(alpha=0.5, gamma=0.0)
    print(f"  Focal Loss γ=0 : {fl_ce(logits, targets).item():.4f}  (≈ BCE)")

    print("\n  ✅ Fonctions de perte opérationnelles !")
