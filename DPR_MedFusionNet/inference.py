"""
inference.py
============
Module d'inférence pour MedFusionNet.

Fonctionnalités :
    1. Grad-CAM : Heatmap d'explicabilité L^c via gradients de la couche de fusion
    2. MC-Dropout : Estimation d'incertitude via N passes stochastiques
    3. Inférence complète sur une image ou un batch

Formules :
    Grad-CAM :
        α^c_k = (1/Z) ΣΣ ∂y^c/∂A^k_{ij}   (poids de classe)
        L^c = ReLU(Σ_k α^c_k · A^k)        (heatmap pondérée)

    MC-Dropout :
        p̄ = (1/N) Σ_{n=1}^{N} f_θ_n(x)    (moyenne des passes)
        u = (1/N) Σ_{n=1}^{N} (f_θ_n(x) - p̄)²  (variance ≈ incertitude)

Usage :
    from inference import MedFusionInference
    engine = MedFusionInference("checkpoints/best.pth")
    result = engine.predict("image.jpg")
    engine.visualize(result, "output.png")
"""

import os
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from model        import MedFusionNet
from preprocessing import CXRPreprocessor


# ═══════════════════════════════════════════════════════════════════════════════
# 1. GRAD-CAM
# ═══════════════════════════════════════════════════════════════════════════════
class GradCAM:
    """
    Grad-CAM (Gradient-weighted Class Activation Mapping).

    Enregistre les activations et gradients de la couche cible via des hooks PyTorch.

    Formule :
        α^c_k = (1/Z) ΣΣ ∂y^c/∂A^k_{ij}    → Importance du k-ème filtre
        L^c   = ReLU(Σ_k α^c_k · A^k)       → Heatmap finale (H, W)

    Référence : Selvaraju et al., 2017 — "Grad-CAM: Visual Explanations from Deep Networks"
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        """
        Args:
            model       : Modèle PyTorch
            target_layer: Couche sur laquelle calculer Grad-CAM
                          (ex: model.gated_fusion.refine)
        """
        self.model        = model
        self.target_layer = target_layer

        # Stockage des activations et gradients via hooks
        self._activations: Optional[torch.Tensor] = None
        self._gradients:   Optional[torch.Tensor] = None

        # Enregistrement des hooks
        self._forward_hook  = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Hook forward : sauvegarde les activations A (B, C, H, W)."""
        self._activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Hook backward : sauvegarde ∂L/∂A (B, C, H, W)."""
        self._gradients = grad_output[0].detach()

    def compute(
        self,
        logit: torch.Tensor,    # (B, 1) — logit brut (non-sigmoïdé)
        target_class: int = 1,  # 1 = pneumonie
    ) -> torch.Tensor:
        """
        Calcule la heatmap Grad-CAM.

        Args:
            logit       : Sortie du modèle (B, 1), DOIT avoir retain_graph=False
            target_class: Classe cible (1 = pneumonie, 0 = normal)

        Returns:
            cam: Heatmap (B, H, W) normalisée dans [0, 1]
        """
        # Gradient de la classe cible par rapport aux activations
        score = logit[:, 0] if target_class == 1 else -logit[:, 0]
        score.sum().backward(retain_graph=True)

        # α^c_k = GAP(∂L/∂A^k)    (B, C)
        gradients   = self._gradients        # (B, C, H, W)
        activations = self._activations      # (B, C, H, W)

        alpha = gradients.mean(dim=[2, 3])   # (B, C)

        # L^c = ReLU(Σ_k α^c_k · A_k)
        # Pondération : (B, C, 1, 1) * (B, C, H, W) → sum sur C → (B, H, W)
        weights = alpha.unsqueeze(-1).unsqueeze(-1)   # (B, C, 1, 1)
        cam     = (weights * activations).sum(dim=1)  # (B, H, W)
        cam     = F.relu(cam)                         # Garder le positif uniquement

        # Normalisation dans [0, 1] par élément de batch
        B, H, W = cam.shape
        cam_flat = cam.view(B, -1)                    # (B, H*W)
        cam_min  = cam_flat.min(dim=1)[0].view(B, 1, 1)
        cam_max  = cam_flat.max(dim=1)[0].view(B, 1, 1)
        cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)

        return cam_norm                               # (B, H, W)

    def remove_hooks(self):
        """Libère les hooks pour éviter les fuites mémoire."""
        self._forward_hook.remove()
        self._backward_hook.remove()


# ═══════════════════════════════════════════════════════════════════════════════
# 2. MC-DROPOUT
# ═══════════════════════════════════════════════════════════════════════════════
class MCDropout:
    """
    Estimation d'incertitude bayésienne approchée par MC-Dropout.

    Effectue N passes forward avec le Dropout activé et calcule
    la moyenne (p̄) et la variance (u) des prédictions.

    Formules :
        p̄ = (1/N) Σ_n p_n             → Prédiction moyenne
        u = (1/N) Σ_n (p_n - p̄)²     → Incertitude épistémique

    Interprétation :
        u ≈ 0   → Modèle confiant
        u > 0.1 → Cas ambigu, révision humaine recommandée
    """

    def __init__(self, model: MedFusionNet, n_passes: int = 20):
        """
        Args:
            model   : MedFusionNet (doit avoir des couches Dropout)
            n_passes: Nombre de passes MC (défaut: 20, recommandé: 30-50)
        """
        self.model    = model
        self.n_passes = n_passes

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Inférence MC-Dropout.

        Args:
            x: Batch d'images (B, 3, H, W)

        Returns:
            Dictionnaire :
                'mean'        : p̄ ∈ [0, 1]  — Probabilité moyenne (B,)
                'uncertainty' : u ∈ [0, 1]  — Incertitude (B,)
                'std'         : σ            — Écart-type des passes (B,)
                'all_probs'   : Toutes les prédictions (N, B)
        """
        # Activer le Dropout en mode éval
        self.model.enable_mc_dropout()

        all_probs = []
        for _ in range(self.n_passes):
            out = self.model(x)
            probs = out["prob"].squeeze(1)   # (B,)
            all_probs.append(probs)

        all_probs = torch.stack(all_probs)   # (N, B)

        mean        = all_probs.mean(dim=0)  # (B,) — p̄
        variance    = all_probs.var(dim=0)   # (B,) — u (incertitude)
        std         = all_probs.std(dim=0)   # (B,) — σ

        return {
            "mean"       : mean,
            "uncertainty": variance,
            "std"        : std,
            "all_probs"  : all_probs,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 3. MOTEUR D'INFÉRENCE COMPLET
# ═══════════════════════════════════════════════════════════════════════════════
class MedFusionInference:
    """
    Interface d'inférence complète pour MedFusionNet.

    Combine :
        - Prédiction (probabilité de pneumonie)
        - Grad-CAM (heatmap d'explicabilité)
        - MC-Dropout (estimation d'incertitude)

    Exemple d'usage :
        engine = MedFusionInference("checkpoints/best.pth")
        result = engine.predict("path/to/chest_xray.jpg")
        engine.visualize(result, save_path="outputs/result.png")
    """

    # Seuils d'interprétation clinique
    UNCERTAINTY_THRESHOLD = 0.10   # Au-dessus → révision recommandée
    PROB_THRESHOLD        = 0.50   # Seuil de classification

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        img_size: int = 384,
        mc_passes: int = 20,
    ):
        """
        Args:
            checkpoint_path: Chemin vers le fichier .pth du meilleur modèle
            device         : Dispositif cible (None → auto-détection)
            img_size       : Taille d'image attendue par le modèle
            mc_passes      : Nombre de passes MC-Dropout
        """
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        # ── Chargement du modèle ─────────────────────────────────────────────
        self.model = MedFusionNet(pretrained=False).to(self.device)

        if Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            state_dict = ckpt.get("model_state", ckpt)
            # Gérer les clés DataParallel ("module." prefix)
            state_dict = {
                k.replace("module.", ""): v for k, v in state_dict.items()
            }
            self.model.load_state_dict(state_dict)
            print(f"[INFO] Checkpoint chargé : {checkpoint_path}")
        else:
            print(f"[WARN] Checkpoint introuvable : {checkpoint_path} — poids aléatoires")

        self.model.eval()

        # ── Composants ───────────────────────────────────────────────────────
        self.preprocessor = CXRPreprocessor(img_size=img_size)
        self.gradcam      = GradCAM(self.model, self.model.gated_fusion.refine)
        self.mc_dropout   = MCDropout(self.model, n_passes=mc_passes)

        self.img_size = img_size
        print(f"[INFO] MedFusionInference prêt | Device: {self.device} | "
              f"MC passes: {mc_passes}")

    def predict(
        self,
        image: str | Image.Image | torch.Tensor,
        compute_cam: bool = True,
        compute_uncertainty: bool = True,
    ) -> Dict:
        """
        Inférence complète sur une image.

        Args:
            image              : Chemin, PIL.Image ou Tensor (1, 3, H, W)
            compute_cam        : Calculer la Grad-CAM heatmap
            compute_uncertainty: Calculer l'incertitude MC-Dropout

        Returns:
            Dictionnaire contenant :
                'prob'        : float — Probabilité de pneumonie ∈ [0, 1]
                'prediction'  : str   — 'PNEUMONIA' ou 'NORMAL'
                'uncertainty' : float — Variance MC-Dropout u ∈ [0, 1]
                'confidence'  : float — 1 - u (confiance)
                'cam_heatmap' : ndarray (H, W) — Heatmap Grad-CAM normalisée
                'gate_value'  : float — Valeur moyenne du gate g
                'review_flag' : bool  — True si révision humaine recommandée
        """
        # ── Prétraitement ────────────────────────────────────────────────────
        if isinstance(image, (str, Path)):
            tensor = self.preprocessor(str(image)).to(self.device)  # (1, 3, H, W)
        elif isinstance(image, Image.Image):
            tensor = self.preprocessor.transform(image).unsqueeze(0).to(self.device)
        else:
            tensor = image.to(self.device)

        result = {}

        # ── Inférence principale + Grad-CAM ──────────────────────────────────
        self.model.eval()
        tensor.requires_grad_(compute_cam)

        out   = self.model(tensor)
        prob  = out["prob"].squeeze().item()
        pred  = "PNEUMONIA" if prob >= self.PROB_THRESHOLD else "NORMAL"
        gate  = out["gate"].mean().item()

        result["prob"]       = prob
        result["prediction"] = pred
        result["gate_value"] = gate

        # ── Grad-CAM ─────────────────────────────────────────────────────────
        if compute_cam:
            cam = self.gradcam.compute(
                out["logit"],
                target_class=1 if pred == "PNEUMONIA" else 0
            )   # (1, H, W) → (H, W)
            cam_np = cam.squeeze().cpu().numpy()   # (h, w) en basse résolution

            # Upsampling vers la résolution de l'image d'entrée
            cam_upsampled = F.interpolate(
                cam.unsqueeze(1),                    # (1, 1, h, w)
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze().cpu().numpy()                # (H, W)

            result["cam_heatmap"] = cam_upsampled    # (384, 384)
            result["cam_small"]   = cam_np           # (12, 12)
        else:
            result["cam_heatmap"] = None
            result["cam_small"]   = None

        # ── MC-Dropout ───────────────────────────────────────────────────────
        if compute_uncertainty:
            with torch.no_grad():
                mc_out = self.mc_dropout.predict(tensor)

            uncertainty = mc_out["uncertainty"].squeeze().item()
            result["uncertainty"] = uncertainty
            result["confidence"]  = 1.0 - float(np.clip(uncertainty * 10, 0, 1))
            result["mc_mean"]     = mc_out["mean"].squeeze().item()
            result["mc_std"]      = mc_out["std"].squeeze().item()
        else:
            result["uncertainty"] = 0.0
            result["confidence"]  = 1.0

        # ── Flag de révision ─────────────────────────────────────────────────
        result["review_flag"] = (
            result["uncertainty"] > self.UNCERTAINTY_THRESHOLD or
            0.3 < prob < 0.7   # Zone de doute
        )

        return result

    def visualize(
        self,
        result: Dict,
        original_image: Optional[str | Image.Image] = None,
        save_path: Optional[str] = None,
        show: bool = True,
    ) -> None:
        """
        Visualise les résultats d'inférence (image, Grad-CAM, métriques).

        Args:
            result        : Sortie de predict()
            original_image: Image originale (chemin ou PIL.Image)
            save_path     : Chemin de sauvegarde de la figure
            show          : Afficher la figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(
            f"MedFusionNet — Prédiction : {result['prediction']}  "
            f"(p = {result['prob']:.3f})",
            fontsize=14, fontweight="bold",
        )

        # ── 1. Image originale ────────────────────────────────────────────────
        ax0 = axes[0]
        if original_image is not None:
            if isinstance(original_image, str):
                img = Image.open(original_image).convert("RGB")
            else:
                img = original_image
            ax0.imshow(img, cmap="gray")
        ax0.set_title("Radiographie originale")
        ax0.axis("off")

        # ── 2. Grad-CAM ───────────────────────────────────────────────────────
        ax1 = axes[1]
        if result["cam_heatmap"] is not None:
            cam_colored = cm.jet(result["cam_heatmap"])[:, :, :3]  # RGB

            if original_image is not None:
                img_np = np.array(img.resize((self.img_size, self.img_size)))
                img_np = img_np / 255.0
                overlay = 0.5 * img_np + 0.5 * cam_colored
                ax1.imshow(np.clip(overlay, 0, 1))
            else:
                ax1.imshow(cam_colored)

        ax1.set_title(
            f"Grad-CAM L^c\n(Gate g = {result['gate_value']:.3f})"
        )
        ax1.axis("off")

        # ── 3. Estimation d'incertitude ───────────────────────────────────────
        ax2 = axes[2]
        u   = result.get("uncertainty", 0.0)
        p   = result["prob"]
        std = result.get("mc_std", 0.0)

        # Jauge circulaire
        theta = np.linspace(0, 2 * np.pi, 200)
        ax2.plot(np.cos(theta), np.sin(theta), "k-", lw=1.5, alpha=0.3)

        # Probabilité (arc vert/rouge)
        theta_p = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi * p, 200)
        color   = "#e74c3c" if p >= 0.5 else "#2ecc71"
        ax2.plot(np.cos(theta_p), np.sin(theta_p), "-", color=color, lw=5)

        # Labels
        ax2.text(0, 0.2, f"p = {p:.3f}", ha="center", va="center",
                 fontsize=16, fontweight="bold", color=color)
        ax2.text(0, -0.15, f"u = {u:.4f}", ha="center", va="center",
                 fontsize=11, color="gray")
        ax2.text(0, -0.35, f"σ = {std:.4f}", ha="center", va="center",
                 fontsize=10, color="gray")

        flag_txt = "⚠️ Révision recommandée" if result["review_flag"] else "✅ Décision confidente"
        flag_col = "#f39c12" if result["review_flag"] else "#27ae60"
        ax2.text(0, -0.6, flag_txt, ha="center", va="center",
                 fontsize=10, color=flag_col, fontweight="bold")

        ax2.set_xlim(-1.3, 1.3)
        ax2.set_ylim(-0.9, 1.3)
        ax2.set_aspect("equal")
        ax2.set_title("MC-Dropout — Incertitude")
        ax2.axis("off")

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"[INFO] Figure sauvegardée : {save_path}")

        if show:
            plt.show()
        else:
            plt.close(fig)

    def predict_batch(
        self,
        image_paths: List[str],
        output_csv: Optional[str] = None,
    ) -> List[Dict]:
        """
        Inférence en lot sur une liste d'images.

        Args:
            image_paths: Liste de chemins vers les images
            output_csv : Chemin pour sauvegarder les résultats (CSV)

        Returns:
            Liste de dictionnaires de résultats
        """
        import csv

        results = []
        for i, path in enumerate(image_paths):
            print(f"  [{i+1}/{len(image_paths)}] {Path(path).name}")
            try:
                res = self.predict(path, compute_cam=False, compute_uncertainty=True)
                res["image_path"] = path
                results.append(res)
            except Exception as e:
                print(f"  [ERREUR] {path}: {e}")

        if output_csv:
            keys = ["image_path", "prediction", "prob", "uncertainty",
                    "confidence", "gate_value", "review_flag"]
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(results)
            print(f"[INFO] Résultats sauvegardés : {output_csv}")

        return results

    def __del__(self):
        """Libère les hooks Grad-CAM."""
        try:
            self.gradcam.remove_hooks()
        except Exception:
            pass


# ─── Test rapide ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("  MedFusionNet — Test du module d'inférence")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Moteur avec poids aléatoires (pas de checkpoint)
    engine = MedFusionInference(
        checkpoint_path = "checkpoints/best.pth",  # Peut ne pas exister
        device          = device,
        mc_passes       = 5,   # Peu de passes pour le test
    )

    # Image synthétique
    dummy_img = Image.fromarray(
        np.random.randint(50, 200, (512, 512), dtype=np.uint8)
    ).convert("RGB")

    print("\n  Inférence sur image synthétique...")
    result = engine.predict(dummy_img, compute_cam=True, compute_uncertainty=True)

    print(f"\n  ─── Résultats ───────────────────────────────")
    print(f"  Prédiction   : {result['prediction']}")
    print(f"  Probabilité  : p = {result['prob']:.4f}")
    print(f"  Incertitude  : u = {result['uncertainty']:.4f}")
    print(f"  Confiance    : c = {result['confidence']:.4f}")
    print(f"  Gate value   : g = {result['gate_value']:.4f}")
    print(f"  Grad-CAM     : shape = {result['cam_heatmap'].shape}")
    print(f"  Révision     : {'⚠️  Oui' if result['review_flag'] else '✅ Non'}")

    # Visualisation sans affichage (mode headless)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        engine.visualize(result, original_image=dummy_img,
                         save_path=tmp.name, show=False)
        print(f"\n  Visualisation sauvegardée : {tmp.name}")

    print("\n  ✅ Module d'inférence opérationnel !")
