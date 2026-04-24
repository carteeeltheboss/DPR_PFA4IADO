"""
scenes/scene_03_architecture.py
Scene 3 — MedFusionNet Architecture
Covers: input → DenseNet-121 branch + Swin Transformer branch → concat fusion head.
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


# ─── helper: feature map grid ────────────────────────────────────────────────
def make_feature_map(rows=4, cols=4, cell_size=0.22,
                     color=TEAL, label="") -> VGroup:
    cells = VGroup()
    for r in range(rows):
        for c in range(cols):
            rect = Square(side_length=cell_size)
            brightness = 0.2 + 0.6 * ((r + c) % 3) / 2.0
            rect.set_fill(color, opacity=brightness)
            rect.set_stroke(BG, width=0.5)
            rect.move_to(RIGHT * c * cell_size + DOWN * r * cell_size)
            cells.add(rect)
    lbl = Text(label, font="Monospace", font_size=13, color=color)
    lbl.next_to(cells, DOWN, buff=0.12)
    return VGroup(cells, lbl)


class Scene03Architecture(Scene):
    """Full two-branch architecture diagram."""

    def construct(self):
        self.camera.background_color = BG

        # ── Title ──────────────────────────────────────────────────────────────
        title = make_section_title("Phase 3 — MedFusionNet Architecture")
        self.play(FadeIn(title))
        self.wait(1.0)
        self.play(FadeOut(title))

        # ── Input image ────────────────────────────────────────────────────────
        input_box = make_block("Input\n3 × 224 × 224", width=2.4, height=1.2,
                               fill=PANEL, color=WHITE)
        input_box.move_to(LEFT * 5.5)

        input_lbl = small_text("Chest X-ray tensor")
        input_lbl.next_to(input_box, DOWN, buff=0.15)

        self.play(FadeIn(input_box), FadeIn(input_lbl))

        # ── Branch split ───────────────────────────────────────────────────────
        split_dot = Dot(color=WHITE, radius=0.07)
        split_dot.move_to(input_box[0].get_right() + RIGHT * 0.6)

        arr_to_split = make_arrow(input_box[0].get_right(), split_dot.get_center())
        self.play(GrowArrow(arr_to_split), FadeIn(split_dot))

        # Upper branch arrow (DenseNet)
        arr_dense = make_arrow(split_dot.get_center(),
                               split_dot.get_center() + RIGHT * 1.2 + UP * 1.8,
                               color=DENSE_COLOR)
        # Lower branch arrow (Swin)
        arr_swin  = make_arrow(split_dot.get_center(),
                               split_dot.get_center() + RIGHT * 1.2 + DOWN * 1.8,
                               color=SWIN_COLOR)

        self.play(GrowArrow(arr_dense), GrowArrow(arr_swin))

        # ── DenseNet-121 branch ────────────────────────────────────────────────
        dense_title = Text("DenseNet-121", font="Monospace",
                           font_size=20, color=DENSE_COLOR, weight=BOLD)
        dense_title.move_to(arr_dense.get_end() + RIGHT * 0.1 + UP * 0.35)

        # DenseBlocks
        dense_blocks_data = [
            ("DenseBlock 1\n(64 ch)", 1.8, 0.75),
            ("DenseBlock 2\n(128 ch)", 1.8, 0.75),
            ("DenseBlock 3\n(256 ch)", 1.8, 0.75),
            ("DenseBlock 4\n(512 ch)", 1.8, 0.75),
        ]
        dense_chain = VGroup()
        prev_right = arr_dense.get_end() + RIGHT * 0.05
        for lbl, w, h in dense_blocks_data:
            blk = make_block(lbl, width=w, height=h,
                              fill=PANEL, color=DENSE_COLOR,
                              text_color=DENSE_COLOR)
            blk.move_to(prev_right + RIGHT * (w / 2 + 0.15) + UP * 1.8)
            dense_chain.add(blk)
            prev_right = blk[0].get_right()

        dense_arrows = VGroup()
        for i in range(len(dense_chain) - 1):
            a = make_arrow(dense_chain[i][0].get_right(),
                           dense_chain[i + 1][0].get_left(),
                           color=DENSE_COLOR)
            dense_arrows.add(a)

        # GAP output
        gap_dense = make_block("GAP\n1024-d", width=1.5, height=0.75,
                               fill=PANEL, color=DENSE_COLOR)
        gap_dense.next_to(dense_chain[-1], RIGHT, buff=0.3)
        arr_gap_d = make_arrow(dense_chain[-1][0].get_right(),
                               gap_dense[0].get_left(), color=DENSE_COLOR)

        self.play(FadeIn(dense_title))
        for blk, *a in zip(dense_chain, [None] + list(dense_arrows)):
            if a[0]:
                self.play(GrowArrow(a[0]), FadeIn(blk), run_time=0.35)
            else:
                self.play(FadeIn(blk), run_time=0.35)
        self.play(GrowArrow(arr_gap_d), FadeIn(gap_dense))

        # ── Swin Transformer branch ────────────────────────────────────────────
        swin_title = Text("Swin Transformer (Tiny)", font="Monospace",
                          font_size=20, color=SWIN_COLOR, weight=BOLD)
        swin_title.move_to(arr_swin.get_end() + RIGHT * 0.1 + DOWN * 1.5)

        swin_stages_data = [
            ("Patch Partition\n4×4 patches", 1.9, 0.75),
            ("Stage 1\nWindow Attn", 1.9, 0.75),
            ("Stage 2\nShifted Window", 1.9, 0.75),
            ("Stage 3-4\nDeep Context", 1.9, 0.75),
        ]
        swin_chain = VGroup()
        prev_right_s = arr_swin.get_end() + RIGHT * 0.05
        for lbl, w, h in swin_stages_data:
            blk = make_block(lbl, width=w, height=h,
                              fill=PANEL, color=SWIN_COLOR,
                              text_color=SWIN_COLOR)
            blk.move_to(prev_right_s + RIGHT * (w / 2 + 0.15) + DOWN * 1.8)
            swin_chain.add(blk)
            prev_right_s = blk[0].get_right()

        swin_arrows = VGroup()
        for i in range(len(swin_chain) - 1):
            a = make_arrow(swin_chain[i][0].get_right(),
                           swin_chain[i + 1][0].get_left(),
                           color=SWIN_COLOR)
            swin_arrows.add(a)

        gap_swin = make_block("Flatten\n768-d", width=1.5, height=0.75,
                              fill=PANEL, color=SWIN_COLOR)
        gap_swin.next_to(swin_chain[-1], RIGHT, buff=0.3)
        arr_gap_s = make_arrow(swin_chain[-1][0].get_right(),
                               gap_swin[0].get_left(), color=SWIN_COLOR)

        self.play(FadeIn(swin_title))
        for blk, *a in zip(swin_chain, [None] + list(swin_arrows)):
            if a[0]:
                self.play(GrowArrow(a[0]), FadeIn(blk), run_time=0.35)
            else:
                self.play(FadeIn(blk), run_time=0.35)
        self.play(GrowArrow(arr_gap_s), FadeIn(gap_swin))

        self.wait(0.8)

        # ── Convergence: Concat ────────────────────────────────────────────────
        concat_x = gap_dense[0].get_right()[0] + 1.2
        concat_y = 0.0

        concat_box = make_block("Concat\n1792-d", width=1.8, height=1.0,
                                fill=PANEL, color=BRAND, text_color=BRAND_LIGHT)
        concat_box.move_to(RIGHT * concat_x + UP * concat_y)

        # Arrows from both GAP outputs to concat
        arr_to_cat_d = make_arrow(gap_dense[0].get_right(),
                                  concat_box[0].get_left() + UP * 0.25,
                                  color=DENSE_COLOR)
        arr_to_cat_s = make_arrow(gap_swin[0].get_right(),
                                  concat_box[0].get_left() + DOWN * 0.25,
                                  color=SWIN_COLOR)

        self.play(GrowArrow(arr_to_cat_d), GrowArrow(arr_to_cat_s),
                  FadeIn(concat_box))

        # ── Fusion MLP head ────────────────────────────────────────────────────
        fusion_steps = [
            ("Linear 512\n+ BN + ReLU", BRAND),
            ("Dropout 0.4", ORANGE),
            ("Linear 128\n+ BN + ReLU", BRAND),
            ("Dropout 0.2", ORANGE),
            ("Linear 2\n(logits)", GREEN),
        ]
        head_chain = VGroup()
        prev = concat_box
        for lbl, color in fusion_steps:
            blk = make_block(lbl, width=1.8, height=0.7,
                              fill=PANEL, color=color)
            blk.next_to(prev, RIGHT, buff=0.25)
            head_chain.add(blk)
            prev = blk

        head_arrows = VGroup()
        nodes = [concat_box] + list(head_chain)
        for i in range(len(nodes) - 1):
            a = make_arrow(nodes[i][0].get_right(),
                           nodes[i + 1][0].get_left(), color=BRAND)
            head_arrows.add(a)

        for blk, arr in zip(head_chain, head_arrows):
            self.play(GrowArrow(arr), FadeIn(blk), run_time=0.38)

        # ── Softmax output ─────────────────────────────────────────────────────
        out_box = make_block("Softmax\nNORMAL / PNEUMONIA",
                             width=2.4, height=1.0,
                             fill=PANEL, color=GREEN, text_color=GREEN)
        out_box.next_to(head_chain[-1], RIGHT, buff=0.3)
        arr_out = make_arrow(head_chain[-1][0].get_right(),
                             out_box[0].get_left(), color=GREEN)
        self.play(GrowArrow(arr_out), FadeIn(out_box))
        self.wait(1.5)

        # ── Dimension annotation ───────────────────────────────────────────────
        dim_ann = VGroup(
            small_text("Dense: 1024-d  |  Swin: 768-d  |  Fused: 1792-d",
                       color=MUTED),
        )
        dim_ann.move_to(DOWN * 3.4)
        self.play(FadeIn(dim_ann))
        self.wait(2.0)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.3)
