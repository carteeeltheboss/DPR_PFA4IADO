"""
scenes/scene_05_training.py
Scene 5 — Training Loop & Loss Function
Covers: mini-batch, forward pass, CrossEntropyLoss with class weights,
        label smoothing, gradient clipping, AdamW, cosine scheduler,
        loss curve decreasing over epochs.
Duration: ~90 s
"""
from __future__ import annotations
from manim import *
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from utils.styles import *


class Scene05Training(Scene):
    """Animated training loop + loss function breakdown."""

    def construct(self):
        self.camera.background_color = BG

        # ── Title ──────────────────────────────────────────────────────────────
        title = make_section_title("Phase 5 — Training Loop")
        self.play(FadeIn(title))
        self.wait(0.9)
        self.play(FadeOut(title))

        # ── 1. Hyperparameter table ────────────────────────────────────────────
        hyp_title = Text("Hyperparameters", font="Monospace",
                         font_size=22, color=WHITE)
        hyp_title.move_to(UP * 3.0)
        self.play(Write(hyp_title))

        params = [
            ("Optimizer",       "AdamW",                   BRAND_LIGHT),
            ("LR (backbone)",   "1e-5",                    TEAL),
            ("LR (head)",       "1e-3",                    TEAL),
            ("Batch size",      "16",                      WHITE),
            ("Epochs",          "40  (early stop pat=8)", WHITE),
            ("Scheduler",       "CosineAnnealingWarmRestarts(T₀=10)", PURPLE),
            ("Grad clip",       "max_norm = 1.0",          ORANGE),
            ("Label smoothing", "ε = 0.1",                 YELLOW),
            ("Class weights",   "NORMAL×1.94 / PNEUMONIA×0.67", RED),
        ]

        rows = VGroup()
        for i, (key, val, color) in enumerate(params):
            key_txt = Text(f"{key:<22}", font="Monospace",
                           font_size=16, color=MUTED)
            val_txt = Text(val, font="Monospace", font_size=16, color=color)
            key_txt.move_to(LEFT * 3.8 + UP * (2.2 - i * 0.48))
            val_txt.next_to(key_txt, RIGHT, buff=0.2)
            rows.add(VGroup(key_txt, val_txt))

        for row in rows:
            self.play(FadeIn(row, shift=RIGHT * 0.15), run_time=0.18)
        self.wait(1.2)
        self.play(FadeOut(rows), FadeOut(hyp_title))

        # ── 2. Training loop cycle ─────────────────────────────────────────────
        loop_title = Text("One Training Iteration", font="Monospace",
                          font_size=22, color=WHITE)
        loop_title.move_to(UP * 3.2)
        self.play(Write(loop_title))

        steps = [
            ("1. Sample mini-batch\n   (16 X-rays + labels)",   LEFT * 4.5 + UP * 1.4,   TEAL),
            ("2. Forward pass\n   model(x) → logits",           LEFT * 1.5 + UP * 1.4,   BRAND_LIGHT),
            ("3. Compute loss\n   CrossEntropy(logits, y)",      RIGHT * 1.5 + UP * 1.4,  YELLOW),
            ("4. Backward pass\n   loss.backward()",             RIGHT * 4.5 + UP * 1.4,  ORANGE),
            ("5. Grad clip\n   max_norm=1.0",                    RIGHT * 4.5 + DOWN * 1.0, RED),
            ("6. optimizer.step()\n   scheduler.step()",         LEFT * 1.5 + DOWN * 1.0, GREEN),
            ("7. Log metrics\n   AUC, F1, Recall",               LEFT * 4.5 + DOWN * 1.0, PURPLE),
        ]

        step_mobs = VGroup()
        arrows_cycle = VGroup()

        for i, (label, pos, color) in enumerate(steps):
            blk = make_block(label, width=2.8, height=1.1,
                             fill=PANEL, color=color)
            blk.move_to(pos)
            step_mobs.add(blk)

        # Arrows connecting steps in a cycle
        positions = [steps[i][1] for i in range(len(steps))]
        for i in range(len(positions)):
            next_i = (i + 1) % len(positions)
            start = step_mobs[i][0].get_right() if i < 4 else step_mobs[i][0].get_left()
            end   = step_mobs[next_i][0].get_left() if next_i < 4 else step_mobs[next_i][0].get_right()
            a = make_arrow(start, end, color=MUTED)
            arrows_cycle.add(a)

        for blk, arr in zip(step_mobs, arrows_cycle):
            self.play(FadeIn(blk, scale=0.9), GrowArrow(arr), run_time=0.38)
        self.wait(1.2)

        # ── 3. Loss function breakdown ─────────────────────────────────────────
        self.play(FadeOut(step_mobs), FadeOut(arrows_cycle), FadeOut(loop_title))

        loss_title = make_section_title("Loss Function: Weighted CrossEntropy + Label Smoothing")
        loss_title.move_to(UP * 3.0).scale(0.78)
        self.play(FadeIn(loss_title))

        # Formula
        formula_bg = make_card(9.5, 1.6)
        formula_bg.move_to(UP * 1.5)

        formula = MathTex(
            r"\mathcal{L} = -\sum_{c} w_c \cdot y_c^{(\epsilon)} \cdot \log \hat{p}_c",
            font_size=34, color=WHITE,
        )
        formula.move_to(formula_bg)
        self.play(FadeIn(formula_bg), Write(formula))

        # Label smoothing annotation
        smooth_ann = MathTex(
            r"y_c^{(\epsilon)} = (1-\epsilon)\, y_c + \frac{\epsilon}{2}, \quad \epsilon=0.1",
            font_size=26, color=YELLOW,
        )
        smooth_ann.next_to(formula_bg, DOWN, buff=0.25)
        self.play(Write(smooth_ann))

        # Weight annotation
        weight_ann = MathTex(
            r"w_{\text{NORMAL}}=1.94, \quad w_{\text{PNEUMONIA}}=0.67",
            font_size=26, color=TEAL,
        )
        weight_ann.next_to(smooth_ann, DOWN, buff=0.2)
        self.play(Write(weight_ann))

        why_smooth = small_text(
            "Label smoothing prevents overconfident predictions (p=100% collapse).",
            color=MUTED,
        )
        why_smooth.next_to(weight_ann, DOWN, buff=0.3)
        self.play(FadeIn(why_smooth))
        self.wait(1.2)

        self.play(FadeOut(Group(*self.mobjects)))

        # ── 4. Loss curve ──────────────────────────────────────────────────────
        curve_title = make_section_title("Training Loss over Epochs")
        self.play(FadeIn(curve_title))
        self.wait(0.5)
        self.play(curve_title.animate.to_edge(UP, buff=0.3).scale(0.8))

        ax = Axes(
            x_range=[0, 40, 5],
            y_range=[0, 0.7, 0.1],
            x_length=9,
            y_length=4.5,
            axis_config={"color": MUTED, "stroke_width": 1.5},
            tips=False,
        ).move_to(DOWN * 0.3)

        ax_labels = ax.get_axis_labels(
            x_label=Text("Epoch", font="Monospace", font_size=16, color=MUTED),
            y_label=Text("Loss",  font="Monospace", font_size=16, color=MUTED),
        )
        self.play(Create(ax), FadeIn(ax_labels))

        # Simulate real training curve from our notebook results
        # Epoch data observed: 0.387, 0.351, 0.333, 0.336, 0.311, 0.310, 0.306, 0.307, 0.301, 0.305
        def train_loss(t):
            return 0.38 * (0.92 ** t) + 0.28 + 0.004 * ((t % 10) - 5) * 0.01

        train_curve = ax.plot(
            lambda t: max(0.05, train_loss(t)),
            x_range=[0, 40, 0.05], color=BRAND, stroke_width=2.5,
        )
        val_curve = ax.plot(
            lambda t: max(0.04, train_loss(t) * 1.15 + 0.04 * abs((t % 10) - 5)),
            x_range=[0, 40, 0.05], color=ORANGE, stroke_width=2.5,
        )

        train_lbl = Text("train loss", font="Monospace",
                         font_size=16, color=BRAND)
        val_lbl   = Text("val loss",   font="Monospace",
                         font_size=16, color=ORANGE)
        train_lbl.next_to(ax, RIGHT, buff=0.2).shift(UP * 0.8)
        val_lbl.next_to(train_lbl, DOWN, buff=0.25)

        self.play(
            Create(train_curve, run_time=2.0),
            Create(val_curve,   run_time=2.0),
            FadeIn(train_lbl), FadeIn(val_lbl),
        )

        # Early stop annotation
        early_line = DashedLine(
            ax.c2p(16, 0), ax.c2p(16, 0.55),
            color=RED, stroke_width=1.5,
        )
        early_lbl = Text("Early stop\n(pat=8)",
                         font="Monospace", font_size=14, color=RED)
        early_lbl.next_to(ax.c2p(16, 0.55), UP, buff=0.08)

        self.play(Create(early_line), FadeIn(early_lbl))
        self.wait(2.0)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.3)
