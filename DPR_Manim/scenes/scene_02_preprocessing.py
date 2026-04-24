"""
scenes/scene_02_preprocessing.py
Scene 2 — Image Preprocessing & Data Augmentation
Covers: resize → ToTensor → Normalize (ImageNet), train augmentations.
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


class Scene02Preprocessing(Scene):
    """Visualise the exact transforms.Compose pipeline from preprocessing.py."""

    def construct(self):
        self.camera.background_color = BG

        # ── 1. Title ───────────────────────────────────────────────────────────
        title = make_section_title("Phase 2 — Image Preprocessing")
        self.play(FadeIn(title, shift=DOWN * 0.2))
        self.wait(1.0)
        self.play(title.animate.to_edge(UP, buff=0.3).scale(0.82))

        # ── 2. Build the pipeline steps ───────────────────────────────────────
        steps = [
            ("Original\nX-ray\n(variable size)", "#1a1a2e", MUTED),
            ("Resize\n(224 × 224)", PANEL, TEAL),
            ("ToTensor\n[0, 255] → [0, 1]", PANEL, BRAND),
            ("Normalize\nImageNet μ/σ", PANEL, PURPLE),
            ("Tensor\n(3 × 224 × 224)", PANEL, GREEN),
        ]

        blocks = VGroup()
        x_positions = [-5.5, -2.8, 0.0, 2.8, 5.5]
        for (lbl, fill, color), x in zip(steps, x_positions):
            blk = make_block(lbl, width=2.3, height=1.5,
                              fill=fill, color=color)
            blk.move_to(RIGHT * x + UP * 0.2)
            blocks.add(blk)

        arrows = VGroup()
        for i in range(len(blocks) - 1):
            a = make_arrow(blocks[i][0].get_right(),
                           blocks[i + 1][0].get_left())
            arrows.add(a)

        for blk, arr in zip(blocks[1:], arrows):
            self.play(FadeIn(blocks[0]) if blk is blocks[1] else FadeIn(blk),
                      run_time=0.01)

        # Animate left-to-right reveal
        self.play(FadeIn(blocks[0]))
        for i in range(len(arrows)):
            self.play(GrowArrow(arrows[i]), FadeIn(blocks[i + 1]), run_time=0.45)
        self.wait(0.8)

        # ── 3. Normalization formula ───────────────────────────────────────────
        formula_bg = make_card(8.5, 1.2)
        formula_bg.move_to(DOWN * 2.4)
        formula = MathTex(
            r"\hat{x} = \frac{x - \mu}{\sigma}",
            r"\quad \mu = (0.485, 0.456, 0.406)",
            r"\quad \sigma = (0.229, 0.224, 0.225)",
            font_size=28, color=WHITE,
        )
        formula.move_to(formula_bg)
        self.play(FadeIn(formula_bg), Write(formula))
        self.wait(1.2)

        # ── 4. Before / after comparison ──────────────────────────────────────
        self.play(FadeOut(blocks), FadeOut(arrows),
                  FadeOut(formula), FadeOut(formula_bg))

        cmp_title = Text("Before  vs  After normalization",
                         font="Monospace", font_size=22, color=MUTED)
        cmp_title.move_to(UP * 2.4)
        self.play(Write(cmp_title))

        # Before: bright grey rectangle (pixel values 0-255 range)
        before = Rectangle(width=2.8, height=2.8)
        before.set_fill("#c0c0c0", opacity=1).set_stroke(BORDER, width=1)
        before_lbl = Text("pixel ≈ 200\n(uint8)", font="Monospace",
                           font_size=16, color="#333")
        before_lbl.move_to(before)
        before_group = VGroup(before, before_lbl)
        before_group.move_to(LEFT * 3.2 + DOWN * 0.3)
        before_tag = small_text("Original pixel values [0, 255]")
        before_tag.next_to(before_group, DOWN, buff=0.18)

        # After: dark rectangle (normalized ~[-2, +2] range)
        after = Rectangle(width=2.8, height=2.8)
        after.set_fill("#1f2937", opacity=1).set_stroke(BRAND, width=1.5)
        after_lbl = Text("value ≈ −0.32\n(float32)", font="Monospace",
                          font_size=16, color=BRAND_LIGHT)
        after_lbl.move_to(after)
        after_group = VGroup(after, after_lbl)
        after_group.move_to(RIGHT * 3.2 + DOWN * 0.3)
        after_tag = small_text("After ImageNet normalization")
        after_tag.next_to(after_group, DOWN, buff=0.18)

        arrow_cmp = make_arrow(before_group.get_right(),
                               after_group.get_left(), color=BRAND)

        self.play(FadeIn(before_group), FadeIn(before_tag))
        self.play(GrowArrow(arrow_cmp))
        self.play(FadeIn(after_group), FadeIn(after_tag))
        self.wait(1.2)

        # ── 5. Augmentation section ───────────────────────────────────────────
        self.play(FadeOut(Group(*self.mobjects)))

        aug_title = make_section_title("Phase 2b — Data Augmentation  (train only)")
        self.play(FadeIn(aug_title, shift=DOWN * 0.2))
        self.wait(0.6)
        self.play(aug_title.animate.to_edge(UP, buff=0.3).scale(0.82))

        augs = [
            ("RandomCrop\n224×224",    TEAL,   "Removes constant borders"),
            ("HorizontalFlip",         BRAND,  "Doubles left/right examples"),
            ("RandomRotation\n±10°",   PURPLE, "Handles patient tilt"),
            ("ColorJitter\nbright/contrast ±0.2", ORANGE,
             "Simulates different scanners"),
        ]

        aug_blocks = VGroup()
        aug_notes  = VGroup()
        for i, (name, color, note) in enumerate(augs):
            blk = make_block(name, width=2.6, height=1.2,
                              fill=PANEL, color=color, text_color=color)
            blk.move_to(LEFT * 4.5 + RIGHT * i * 3.0 + UP * 0.5)
            note_txt = small_text(note)
            note_txt.next_to(blk, DOWN, buff=0.2)
            aug_blocks.add(blk)
            aug_notes.add(note_txt)

        why_note = body_text(
            "Augmentation prevents overfitting on the small NORMAL class.",
            color=YELLOW,
        )
        why_note.move_to(DOWN * 2.5)

        for blk, note_txt in zip(aug_blocks, aug_notes):
            self.play(FadeIn(blk, scale=0.9),
                      FadeIn(note_txt), run_time=0.5)
        self.play(FadeIn(why_note))
        self.wait(2.0)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.3)
