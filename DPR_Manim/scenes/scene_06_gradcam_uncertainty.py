"""
scenes/scene_06_gradcam_uncertainty.py
Scene 6 — Grad-CAM Explainability + MC-Dropout Uncertainty
Duration: ~80 s
"""
from __future__ import annotations
from manim import *
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from utils.styles import *


def make_xray_placeholder(opacity_color=YELLOW, show_highlight=False) -> VGroup:
    """Simulated chest X-ray as a Manim VGroup."""
    bg     = Rectangle(width=2.6, height=2.6)
    bg.set_fill("#111827", opacity=1).set_stroke(BORDER, width=1)
    left   = Ellipse(width=0.85, height=1.5).set_fill("#1e3a5f", opacity=1).set_stroke(color=BG, width=0)
    right  = Ellipse(width=0.85, height=1.5).set_fill("#1e3a5f", opacity=1).set_stroke(color=BG, width=0)
    left.move_to(bg.get_center()  + LEFT  * 0.5)
    right.move_to(bg.get_center() + RIGHT * 0.5)
    parts = [bg, left, right]
    if show_highlight:
        spot = Ellipse(width=0.6, height=0.4)
        spot.set_fill(opacity_color, opacity=0.55).set_stroke(color=opacity_color, width=0)
        spot.move_to(bg.get_center() + RIGHT * 0.42 + DOWN * 0.1)
        parts.append(spot)
    return VGroup(*parts)


def make_heatmap_overlay() -> VGroup:
    """Simulated Grad-CAM heatmap as concentric colour ovals."""
    bg = Rectangle(width=2.6, height=2.6)
    bg.set_fill("#111827", opacity=1).set_stroke(BORDER, width=1)

    center = bg.get_center() + RIGHT * 0.42 + DOWN * 0.1

    c1 = Ellipse(width=1.2, height=0.8).set_fill(RED,    opacity=0.45).set_stroke(color=RED,    width=0)
    c2 = Ellipse(width=0.8, height=0.55).set_fill(YELLOW, opacity=0.55).set_stroke(color=YELLOW, width=0)
    c3 = Ellipse(width=0.45, height=0.3).set_fill(WHITE,  opacity=0.6).set_stroke(color=WHITE,  width=0)
    for c in [c1, c2, c3]:
        c.move_to(center)

    return VGroup(bg, c1, c2, c3)


class Scene06GradCAM(Scene):
    """Grad-CAM visualisation and MC-Dropout uncertainty."""

    def construct(self):
        self.camera.background_color = BG

        # ── Title ──────────────────────────────────────────────────────────────
        title = make_section_title("Phase 6 — Grad-CAM Explainability")
        self.play(FadeIn(title))
        self.wait(0.9)
        self.play(FadeOut(title))

        # ── 1. Target layer annotation ─────────────────────────────────────────
        layer_ann = Text(
            "Target layer: densenet.features.denseblock4",
            font="Monospace", font_size=18, color=DENSE_COLOR,
        )
        layer_ann.move_to(UP * 3.1)
        self.play(FadeIn(layer_ann))
        self.wait(0.6)

        # ── 2. Three-panel: original / heatmap / overlay ───────────────────────
        orig  = make_xray_placeholder(show_highlight=True)
        heat  = make_heatmap_overlay()
        over  = make_xray_placeholder(show_highlight=True)

        # Add heatmap on top of overlay
        heat_small = make_heatmap_overlay()
        heat_small.set_opacity(0.55)
        heat_small.move_to(over.get_center())
        overlay_group = VGroup(over, heat_small)

        orig.move_to(LEFT  * 4.5 + UP * 0.1)
        heat.move_to(ORIGIN + UP * 0.1)
        overlay_group.move_to(RIGHT * 4.5 + UP * 0.1)

        # Labels
        lbl_orig = small_text("1. Original X-ray")
        lbl_heat = small_text("2. Grad-CAM heatmap")
        lbl_over = small_text("3. Overlay")
        lbl_orig.next_to(orig,          DOWN, buff=0.2)
        lbl_heat.next_to(heat,          DOWN, buff=0.2)
        lbl_over.next_to(overlay_group, DOWN, buff=0.2)

        panels = [
            (orig,          lbl_orig),
            (heat,          lbl_heat),
            (overlay_group, lbl_over),
        ]
        for mob, lbl in panels:
            self.play(FadeIn(mob, scale=0.9), FadeIn(lbl), run_time=0.6)
        self.wait(0.8)

        # ── 3. Annotate hot spot ───────────────────────────────────────────────
        spot_circle = Circle(radius=0.42, color=RED, stroke_width=2)
        spot_circle.move_to(overlay_group.get_center() + RIGHT * 0.42 + DOWN * 0.1)
        spot_arrow  = make_arrow(
            overlay_group.get_center() + UP * 1.6 + RIGHT * 1.0,
            spot_circle.get_top(),
            color=RED,
        )
        spot_lbl = Text("Model focused here\n(lower right lobe opacity)",
                        font="Monospace", font_size=14, color=RED)
        spot_lbl.next_to(spot_arrow.get_start(), UP, buff=0.1)

        self.play(Create(spot_circle), GrowArrow(spot_arrow), FadeIn(spot_lbl))
        self.wait(1.0)

        # ── 4. Grad-CAM formula ────────────────────────────────────────────────
        formula_bg = make_card(9.0, 1.3)
        formula_bg.move_to(DOWN * 3.1)

        formula = MathTex(
            r"\text{CAM} = \text{ReLU}\!\left(\sum_k \alpha_k^c \cdot A^k\right)",
            r"\quad \alpha_k^c = \frac{1}{Z}\sum_{i,j} \frac{\partial y^c}{\partial A^k_{ij}}",
            font_size=26, color=WHITE,
        )
        formula.move_to(formula_bg)
        self.play(FadeIn(formula_bg), Write(formula))
        self.wait(1.2)

        # ── 5. Disclaimer ──────────────────────────────────────────────────────
        disc = Text(
            "Grad-CAM is an explanation tool — NOT a diagnosis by itself.",
            font="Monospace", font_size=16, color=YELLOW,
        )
        disc.next_to(formula_bg, DOWN, buff=0.22)
        self.play(FadeIn(disc))
        self.wait(1.0)

        self.play(FadeOut(Group(*self.mobjects)))

        # ══════════════════════════════════════════════════════════════════════
        # MC-DROPOUT UNCERTAINTY
        # ══════════════════════════════════════════════════════════════════════
        mc_title = make_section_title("Phase 6b — MC-Dropout Uncertainty")
        self.play(FadeIn(mc_title))
        self.wait(0.8)
        self.play(FadeOut(mc_title))

        # ── 6. Same image → N forward passes ──────────────────────────────────
        mc_idea = body_text(
            "Keep Dropout active at inference.\nRun the same image N times → collect probabilities.",
            color=WHITE,
        )
        mc_idea.move_to(UP * 2.8)
        self.play(FadeIn(mc_idea))

        img_icon = make_xray_placeholder(show_highlight=True)
        img_icon.scale(0.65).move_to(LEFT * 5.0 + UP * 0.3)
        self.play(FadeIn(img_icon))

        # Simulate N=8 forward passes with slightly different probabilities
        probs_sim = [0.97, 0.94, 0.98, 0.91, 0.96, 0.93, 0.99, 0.95]
        pass_mobs = VGroup()
        for i, p in enumerate(probs_sim):
            bar = make_probability_bar(f"Pass {i+1}", p, width=2.8, bar_color=RED)
            bar.move_to(RIGHT * 0.5 + UP * (1.8 - i * 0.52))
            pass_mobs.add(bar)

        arr_mc = make_arrow(img_icon.get_right(),
                            pass_mobs[0].get_left() + LEFT * 0.5,
                            color=MUTED)
        self.play(GrowArrow(arr_mc))
        for bar in pass_mobs:
            self.play(FadeIn(bar, shift=RIGHT * 0.2), run_time=0.22)
        self.wait(0.5)

        # ── 7. Mean ± std ──────────────────────────────────────────────────────
        mean_p = sum(probs_sim) / len(probs_sim)
        import math
        std_p  = math.sqrt(sum((p - mean_p) ** 2 for p in probs_sim) / len(probs_sim))

        stats_bg = make_card(5.0, 1.4)
        stats_bg.move_to(RIGHT * 4.5 + UP * 0.3)

        mean_txt = Text(f"μ  =  {mean_p:.3f}", font="Monospace",
                        font_size=22, color=WHITE)
        std_txt  = Text(f"σ  =  {std_p:.3f}  →  High confidence",
                        font="Monospace", font_size=18, color=GREEN)
        mean_txt.move_to(stats_bg.get_center() + UP * 0.28)
        std_txt.next_to(mean_txt, DOWN, buff=0.2)

        self.play(FadeIn(stats_bg), Write(mean_txt), Write(std_txt))
        self.wait(0.8)

        # ── 8. Low-confidence example ─────────────────────────────────────────
        uncertain_note = small_text(
            "High σ (e.g. > 0.15) → model is uncertain → flag for clinical review.",
            color=YELLOW,
        )
        uncertain_note.move_to(DOWN * 3.0)
        self.play(FadeIn(uncertain_note))
        self.wait(2.0)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.3)
