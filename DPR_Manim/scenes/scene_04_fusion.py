"""
scenes/scene_04_fusion.py
Scene 4 — Feature Fusion & Classification Head
Covers: concatenation of 1024-d and 768-d vectors, MLP head, softmax, threshold.
Duration: ~70 s
"""
from __future__ import annotations
from manim import *
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from utils.styles import *


def make_vector_strip(dim: int, color: str, label: str,
                      bar_height=0.55, max_width=3.5) -> VGroup:
    """Visual representation of a feature vector as coloured bars."""
    n_shown = min(dim, 16)
    bars = VGroup()
    for i in range(n_shown):
        h = bar_height * (0.3 + 0.7 * ((i * 7 + 3) % 10) / 10.0)
        bar = Rectangle(width=max_width / n_shown * 0.82, height=h)
        bar.set_fill(color, opacity=0.7 + 0.3 * (i % 3) / 2)
        bar.set_stroke(color=BG, width=0.4)
        bar.align_to(ORIGIN, DOWN)
        bar.move_to(RIGHT * i * (max_width / n_shown))
        bars.add(bar)
    lbl = Text(f"{label}\n({dim}-d)", font="Monospace",
               font_size=14, color=color)
    lbl.next_to(bars, DOWN, buff=0.18)
    return VGroup(bars, lbl)


class Scene04Fusion(Scene):
    """Visualise concatenation and the MLP classification head."""

    def construct(self):
        self.camera.background_color = BG

        # ── Title ──────────────────────────────────────────────────────────────
        title = make_section_title("Phase 4 — Feature Fusion")
        self.play(FadeIn(title))
        self.wait(0.9)
        self.play(FadeOut(title))

        # ── 1. Show two feature vectors ───────────────────────────────────────
        dense_vec = make_vector_strip(1024, DENSE_COLOR, "DenseNet-121")
        swin_vec  = make_vector_strip(768,  SWIN_COLOR,  "Swin Transformer",
                                      max_width=2.8)

        dense_vec.move_to(LEFT * 3.8 + UP * 1.2)
        swin_vec.move_to(RIGHT * 3.2 + UP * 1.2)

        self.play(FadeIn(dense_vec, shift=LEFT * 0.3))
        self.play(FadeIn(swin_vec,  shift=RIGHT * 0.3))

        local_lbl  = small_text("Local features\n(edges, opacity, texture)")
        global_lbl = small_text("Global features\n(long-range context, patches)")
        local_lbl.next_to(dense_vec, UP, buff=0.2)
        global_lbl.next_to(swin_vec,  UP, buff=0.2)

        self.play(FadeIn(local_lbl), FadeIn(global_lbl))
        self.wait(1.0)

        # ── 2. Concatenation ───────────────────────────────────────────────────
        fused_vec = make_vector_strip(1792, FUSE_COLOR, "Fused vector",
                                      max_width=6.5)
        fused_vec.move_to(DOWN * 0.8)

        arr_d = make_arrow(dense_vec[0].get_bottom() + DOWN * 0.1,
                           fused_vec[0].get_top()    + LEFT * 1.5,
                           color=DENSE_COLOR)
        arr_s = make_arrow(swin_vec[0].get_bottom()  + DOWN * 0.1,
                           fused_vec[0].get_top()    + RIGHT * 1.5,
                           color=SWIN_COLOR)

        cat_label = Text("torch.cat([ f_dense, f_swin ], dim=1)",
                         font="Monospace", font_size=17, color=BRAND_LIGHT)
        cat_label.next_to(fused_vec, UP, buff=0.55)

        self.play(GrowArrow(arr_d), GrowArrow(arr_s))
        self.play(Write(cat_label))
        self.play(FadeIn(fused_vec, shift=UP * 0.2))
        self.wait(1.0)

        # ── 3. Why concatenation (not gated) ──────────────────────────────────
        note = small_text(
            "Note: MedFusionNet uses simple concatenation, not a learned gate.\n"
            "The MLP head then learns which features matter.",
            color=MUTED,
        )
        note.move_to(DOWN * 2.6)
        self.play(FadeIn(note))
        self.wait(1.2)

        self.play(FadeOut(Group(*self.mobjects)))

        # ── 4. MLP Classification Head ────────────────────────────────────────
        head_title = make_section_title("Classification Head — MLP")
        self.play(FadeIn(head_title))
        self.wait(0.7)
        self.play(head_title.animate.to_edge(UP, buff=0.3).scale(0.8))

        head_steps = [
            ("1792-d\nfused vector",  WHITE,   "Input"),
            ("Linear\n→ 512",         BRAND,   "FC1"),
            ("BN + ReLU\nDropout 0.4",ORANGE,  "Reg"),
            ("Linear\n→ 128",         BRAND,   "FC2"),
            ("BN + ReLU\nDropout 0.2",ORANGE,  "Reg"),
            ("Linear\n→ 2",           GREEN,   "Output"),
        ]

        chain = VGroup()
        x = -5.2
        for lbl, color, tag in head_steps:
            blk = make_block(lbl, width=1.85, height=1.2,
                              fill=PANEL, color=color)
            blk.move_to(RIGHT * x + UP * 0.2)
            tag_txt = small_text(tag, color=color)
            tag_txt.next_to(blk, DOWN, buff=0.14)
            chain.add(VGroup(blk, tag_txt))
            x += 2.1

        chain_arrows = VGroup()
        for i in range(len(chain) - 1):
            a = make_arrow(chain[i][0][0].get_right(),
                           chain[i + 1][0][0].get_left(),
                           color=BRAND)
            chain_arrows.add(a)

        for blk, *a in zip(chain, [None] + list(chain_arrows)):
            if a[0]:
                self.play(GrowArrow(a[0]), FadeIn(blk), run_time=0.4)
            else:
                self.play(FadeIn(blk), run_time=0.4)

        # ── 5. Softmax + threshold ─────────────────────────────────────────────
        self.wait(0.5)

        softmax_title = Text("Softmax → Probabilities → Threshold",
                             font="Monospace", font_size=20, color=WHITE)
        softmax_title.move_to(DOWN * 2.1)
        self.play(FadeIn(softmax_title))

        bar_normal = make_probability_bar("NORMAL",    0.02, bar_color=TEAL)
        bar_pneumo = make_probability_bar("PNEUMONIA", 0.98, bar_color=RED)
        bar_normal.move_to(DOWN * 2.85 + LEFT * 1)
        bar_pneumo.next_to(bar_normal, DOWN, buff=0.3)

        self.play(FadeIn(bar_normal), FadeIn(bar_pneumo))
        self.wait(0.8)

        # Decision threshold box
        thresh_box = make_card(5.2, 1.1, fill=PANEL, stroke=GREEN)
        thresh_box.move_to(DOWN * 3.8 + LEFT * 0.5)

        thresh_txt = Text(
            "if P(PNEUMONIA) ≥ 0.5  →  PNEUMONIA\nelse                   →  NORMAL",
            font="Monospace", font_size=17, color=GREEN,
        )
        thresh_txt.move_to(thresh_box)
        self.play(FadeIn(thresh_box), Write(thresh_txt))
        self.wait(2.0)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.3)
