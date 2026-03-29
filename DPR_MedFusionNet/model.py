"""
model.py
========
Architecture MedFusionNet : CNN (DenseNet-121) + Swin Transformer + Gated Fusion.

Flux de données :
    Input (B, 3, 384, 384)
        │
        ├──► Branche Locale  (DenseNet-121) ──► F_c (B, d, h, w)
        │
        └──► Branche Globale (Swin-T)       ──► F_s (B, d, h, w)
                │
                ▼
        Gated Fusion : g = σ(W_g·[GAP(F_c), GAP(F_s)] + b_g)
                       F = g ⊙ F_c + (1-g) ⊙ F_s
                │
                ▼
        GAP → Dropout → Classifier → p (probabilité)

Relations mathématiques :
    g  = σ(W_g · concat(GAP(F_c), GAP(F_s)) + b_g)     [Eq. 1]
    F  = g ⊙ F_c + (1-g) ⊙ F_s                         [Eq. 2]
    p  = σ(W_cls · GAP(F))                               [Eq. 3]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121, DenseNet121_Weights
from typing import Tuple as Tuple_

# Swin Transformer via timm (pip install timm)
try:
    import timm
    HAS_TIMM = True
except ImportError:
    HAS_TIMM = False
    print("[WARN] timm non disponible. Installez-le avec : pip install timm")


# ─── Constantes ────────────────────────────────────────────────────────────────
DENSENET_OUT_CHANNELS = 1024   # Canaux de sortie de DenseNet-121 (features layer)
SWIN_T_OUT_CHANNELS   = 768    # Canaux de sortie de Swin-Tiny (dernière étape)
FUSION_DIM            = 512    # Dimension de la feature fusionnée
DROPOUT_RATE          = 0.3    # Taux de dropout (utilisé pour MC-Dropout)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. BRANCHE LOCALE — DenseNet-121
# ═══════════════════════════════════════════════════════════════════════════════
class LocalBranch(nn.Module):
    """
    Extrait les features locales (textures, opacités) via DenseNet-121.

    Utilise les poids ImageNet pré-entraînés.
    La tête de classification originale est supprimée.
    Sortie : F_c ∈ ℝ^(B × 1024 × 12 × 12) pour une entrée 384×384.
    """

    def __init__(self, pretrained: bool = True, freeze_until: int = 0):
        """
        Args:
            pretrained   : Charger les poids ImageNet
            freeze_until : Geler les `freeze_until` premiers dense blocks
                           (0 = pas de gel, 4 = tout geler sauf la tête)
        """
        super().__init__()

        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = densenet121(weights=weights)

        # Extraire uniquement la partie features (sans le classifier)
        # DenseNet-121 : features → (B, 1024, H/32, W/32)
        self.features = backbone.features   # Module séquentiel complet

        # Normalisation batch finale + activation
        self.bn_final = nn.BatchNorm2d(DENSENET_OUT_CHANNELS)

        # Projection vers la dimension de fusion commune
        self.projection = nn.Sequential(
            nn.Conv2d(DENSENET_OUT_CHANNELS, FUSION_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(FUSION_DIM),
            nn.ReLU(inplace=True),
        )

        # Gel partiel du backbone
        if freeze_until > 0:
            self._freeze_layers(freeze_until)

    def _freeze_layers(self, n_blocks: int):
        """Gèle les n premiers dense blocks de DenseNet."""
        layers_to_freeze = [
            "conv0", "norm0", "relu0", "pool0",
            "denseblock1", "transition1",
            "denseblock2", "transition2",
            "denseblock3", "transition3",
        ][:n_blocks * 2]   # 2 sous-modules par bloc (dense + transition)

        for name, param in self.features.named_parameters():
            if any(name.startswith(layer) for layer in layers_to_freeze):
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 384, 384)

        Returns:
            F_c: (B, FUSION_DIM, h, w)  — feature map locale
        """
        feat = self.features(x)        # (B, 1024, 12, 12)
        feat = self.bn_final(feat)
        feat = F.relu(feat, inplace=True)
        feat = self.projection(feat)   # (B, 512, 12, 12)
        return feat


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BRANCHE GLOBALE — Swin Transformer
# ═══════════════════════════════════════════════════════════════════════════════
class GlobalBranch(nn.Module):
    """
    Capture le contexte global et la symétrie pulmonaire via Swin Transformer.

    Utilise swin_tiny_patch4_window12_384 (entrée native 384×384).
    Sortie : F_s ∈ ℝ^(B × 768 × 12 × 12) pour une entrée 384×384.

    Si timm n'est pas disponible, une version MLP de substitution est utilisée.
    """

    def __init__(self, pretrained: bool = True):
        """
        Args:
            pretrained: Charger les poids ImageNet pré-entraînés (via timm)
        """
        super().__init__()

        if HAS_TIMM:
            # Swin-Tiny optimisé pour 384×384 (via interpolation Swin-V2)
            self.swin = timm.create_model(
                "swinv2_cr_tiny_ns_224.sw_in1k",
                pretrained=pretrained,
                img_size=384,       # timm gère l'interpolation des window sizes
                num_classes=0,      # Supprime la classification head
                global_pool="",     # Désactive le pooling global de timm
            )
            self._use_timm = True
            out_channels = SWIN_T_OUT_CHANNELS   # 768

        else:
            # Fallback : transformer simplifié (pour tests sans timm)
            print("[WARN] Utilisation du GlobalBranch de substitution (MLP).")
            self._use_timm = False
            out_channels = FUSION_DIM

            self.fallback = nn.Sequential(
                nn.AdaptiveAvgPool2d((12, 12)),
                nn.Conv2d(3, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        # Projection vers la dimension commune de fusion
        self.projection = nn.Sequential(
            nn.Conv2d(out_channels, FUSION_DIM, kernel_size=1, bias=False),
            nn.BatchNorm2d(FUSION_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 384, 384)

        Returns:
            F_s: (B, FUSION_DIM, h, w) — feature map globale
        """
        if self._use_timm:
            # Swin output can vary by model: (B, N, C), (B, H, W, C), or (B, C, H, W)
            feat = self.swin.forward_features(x)   # (B, ...)
            
            if feat.dim() == 3:
                # (B, N, C) -> (B, C, H, W)
                B, N, C = feat.shape
                H = W = int(N ** 0.5)
                feat = feat.permute(0, 2, 1).reshape(B, C, H, W)
            elif feat.dim() == 4:
                # Could be (B, H, W, C) or (B, C, H, W)
                # We expect channels to be SWIN_T_OUT_CHANNELS (768)
                if feat.shape[3] == SWIN_T_OUT_CHANNELS:
                    # (B, H, W, C) -> (B, C, H, W)
                    feat = feat.permute(0, 3, 1, 2)
                # else it's already (B, C, H, W)
        else:
            feat = self.fallback(x)               # (B, FUSION_DIM, 12, 12)

        feat = self.projection(feat)               # (B, 512, 12, 12)
        return feat


# ═══════════════════════════════════════════════════════════════════════════════
# 3. GATED FUSION MODULE
# ═══════════════════════════════════════════════════════════════════════════════
class GatedFusion(nn.Module):
    """
    Mécanisme de fusion guidée par un coefficient g appris.

    Formules :
        g = σ( W_g · concat(GAP(F_c), GAP(F_s)) + b_g )   [scalaire par canal]
        F = g ⊙ F_c + (1-g) ⊙ F_s                         [feature fusionnée]

    Le coefficient g est appris via une couche linéaire sur la concaténation
    des représentations globales GAP des deux branches.
    """

    def __init__(self, feat_dim: int = FUSION_DIM):
        """
        Args:
            feat_dim: Dimension des canaux des feature maps (F_c et F_s)
        """
        super().__init__()

        # W_g : (2 * feat_dim) → feat_dim
        # Entrée : concat(GAP(F_c), GAP(F_s)) ∈ ℝ^(B × 2*feat_dim)
        # Sortie : g ∈ ℝ^(B × feat_dim)
        self.gate_fc = nn.Linear(2 * feat_dim, feat_dim, bias=True)

        # Couche de raffinement post-fusion
        self.refine = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        f_c: torch.Tensor,
        f_s: torch.Tensor,
    ) -> Tuple_:
        """
        Args:
            f_c: Feature locale  (B, D, h, w)
            f_s: Feature globale (B, D, h, w)

        Returns:
            f_fused  : Feature fusionnée (B, D, h, w)
            gate_val : Valeur du gate g  (B, D, 1, 1) — pour visualisation
        """
        B, D, H, W = f_c.shape

        # ------------------------------------------------------------------
        # GAP sur chaque branche : (B, D, H, W) → (B, D)
        gap_c = f_c.mean(dim=[2, 3])   # Global Average Pooling
        gap_s = f_s.mean(dim=[2, 3])

        # Concaténation et calcul du gate : g ∈ (0, 1)^D par canal
        # g = σ(W_g · [gap_c | gap_s] + b_g)
        combined = torch.cat([gap_c, gap_s], dim=1)   # (B, 2D)
        g = torch.sigmoid(self.gate_fc(combined))      # (B, D)
        g = g.unsqueeze(-1).unsqueeze(-1)              # (B, D, 1, 1)

        # ------------------------------------------------------------------
        # Fusion pondérée
        # F = g ⊙ F_c + (1-g) ⊙ F_s
        f_fused = g * f_c + (1.0 - g) * f_s          # (B, D, H, W)

        # Raffinement convolutionnel
        f_fused = self.refine(f_fused)                # (B, D, H, W)

        return f_fused, g


# Trick pour annotation de retour sans import Tuple au niveau du module



# ═══════════════════════════════════════════════════════════════════════════════
# 4. ARCHITECTURE COMPLÈTE — MedFusionNet
# ═══════════════════════════════════════════════════════════════════════════════
class MedFusionNet(nn.Module):
    """
    Architecture hybride CNN + Swin Transformer pour la détection de pneumonie.

    Composants :
        - LocalBranch  : DenseNet-121 pré-entraîné → F_c
        - GlobalBranch : Swin Transformer           → F_s
        - GatedFusion  : Fusion adaptative g        → F
        - Classifier   : FC + Dropout → p ∈ [0, 1]

    MC-Dropout :
        En mode entraînement, le Dropout est actif (comportement standard).
        Pour l'inférence MC-Dropout, appeler enable_mc_dropout() puis
        effectuer N passes forward. La variance des sorties estime l'incertitude.

    Grad-CAM :
        La feature map de la couche 'gradcam_layer' est utilisée pour
        calculer les heatmaps d'explicabilité (voir inference.py).
    """

    def __init__(
        self,
        num_classes: int = 1,        # Sortie binaire (1 = pneumonie)
        feat_dim: int = FUSION_DIM,
        dropout_rate: float = DROPOUT_RATE,
        pretrained: bool = True,
        freeze_cnn_blocks: int = 2,  # Geler les 2 premiers dense blocks
    ):
        super().__init__()

        # ── Branches ─────────────────────────────────────────────────────────
        self.local_branch  = LocalBranch(pretrained=pretrained,
                                         freeze_until=freeze_cnn_blocks)
        self.global_branch = GlobalBranch(pretrained=pretrained)

        # ── Fusion ───────────────────────────────────────────────────────────
        self.gated_fusion = GatedFusion(feat_dim=feat_dim)

        # ── Classification head ───────────────────────────────────────────────
        # GAP → Dropout → FC → p
        self.gap       = nn.AdaptiveAvgPool2d(1)      # (B, D, 1, 1) → scalaire
        self.dropout   = nn.Dropout(p=dropout_rate)   # MC-Dropout
        self.classifier = nn.Linear(feat_dim, num_classes)

        # ── Référence de la couche pour Grad-CAM ──────────────────────────────
        # On cible la couche de raffinement de GatedFusion
        self.gradcam_layer = self.gated_fusion.refine  # Conv2d cible

        # ── Initialisation des couches ajoutées ───────────────────────────────
        self._init_weights()

    def _init_weights(self):
        """Initialise les couches FC et de projection avec Xavier uniform."""
        for module in [self.classifier, self.gated_fusion.gate_fc]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def enable_mc_dropout(self):
        """
        Active le Dropout en mode évaluation pour MC-Dropout.
        Appeler avant les N passes forward d'inférence incertaine.
        """
        def _enable_dropout(m):
            if isinstance(m, nn.Dropout):
                m.train()

        self.eval()           # Gèle BN, désactive les gradients
        self.apply(_enable_dropout)   # Réactive les couches Dropout

    def forward(self, x: torch.Tensor) -> dict:
        """
        Passe forward complète.

        Args:
            x: Batch d'images (B, 3, 384, 384)

        Returns:
            Dictionnaire contenant :
                'logit'   : Logit brut        (B, 1)
                'prob'    : Probabilité sigmoid (B, 1)
                'f_fused' : Feature fusionnée  (B, D, h, w)  — pour Grad-CAM
                'gate'    : Valeur du gate g   (B, D, 1, 1)  — pour analyse
                'gap_feat': Feature après GAP  (B, D)        — pour visualisation
        """
        # ── 1. Extraction des features ────────────────────────────────────────
        f_c = self.local_branch(x)    # (B, D, 12, 12)  — textures locales
        f_s = self.global_branch(x)   # (B, D, 12, 12)  — contexte global

        # ── 2. Fusion guidée par le gate ──────────────────────────────────────
        f_fused, gate = self.gated_fusion(f_c, f_s)   # (B, D, 12, 12), (B, D, 1, 1)

        # ── 3. Pooling global et classification ───────────────────────────────
        gap_feat = self.gap(f_fused).flatten(1)        # (B, D)
        gap_feat = self.dropout(gap_feat)              # MC-Dropout
        logit    = self.classifier(gap_feat)           # (B, 1)
        prob     = torch.sigmoid(logit)                # (B, 1) ∈ [0, 1]

        return {
            "logit"   : logit,
            "prob"    : prob,
            "f_fused" : f_fused,
            "f_c"     : f_c,
            "f_s"     : f_s,
            "gate"    : gate,
            "gap_feat": gap_feat,
        }

    def count_parameters(self) -> dict:
        """Retourne le nombre de paramètres par composant."""
        def count(module):
            return sum(p.numel() for p in module.parameters() if p.requires_grad)

        return {
            "local_branch" : count(self.local_branch),
            "global_branch": count(self.global_branch),
            "gated_fusion" : count(self.gated_fusion),
            "classifier"   : count(self.classifier),
            "total"        : count(self),
        }


# ─── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 65)
    print("  MedFusionNet — Test de l'architecture hybride")
    print("=" * 65)

    # Utiliser CPU pour le test (pas besoin de GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device : {device}")

    model = MedFusionNet(pretrained=False)   # False pour tester sans téléchargement
    model = model.to(device)
    model.eval()

    # Batch synthétique
    B = 2
    x = torch.randn(B, 3, 384, 384, device=device)
    print(f"\n  Input  : {x.shape}")

    with torch.no_grad():
        out = model(x)

    print(f"  Output | logit   : {out['logit'].shape}")
    print(f"         | prob    : {out['prob'].shape}  "
          f"(range: [{out['prob'].min():.3f}, {out['prob'].max():.3f}])")
    print(f"         | f_fused : {out['f_fused'].shape}")
    print(f"         | gate    : {out['gate'].shape}")

    # Paramètres
    params = model.count_parameters()
    print(f"\n  Paramètres entraînables :")
    for key, val in params.items():
        print(f"    {key:<20} : {val:>10,}")

    print("\n  ✅ Architecture MedFusionNet opérationnelle !")
