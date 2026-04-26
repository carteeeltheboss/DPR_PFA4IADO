"""
scene_B_training_tensor.py
==========================
PHASE: Training Loop — microscopic detail
Takes ONE real sample (the same 8×8 patch) and walks through:
  1. Tensor representation (3-channel matrix)
  2. Forward pass through DenseNet (show feature map shrinking)
  3. Forward pass through Swin (show patch tokens)
  4. Concatenation
  5. MLP (show numbers flowing through each layer)
  6. Cross-entropy loss with class weights
  7. Backward pass (gradient heat-map)
  8. Weight update (one weight shown changing)
Duration: ~130 s
"""
from __future__ import annotations
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from manim import *
import numpy as np
from utils.styles import *

# ── Fixed data: our 8×8 normalised patch (channel R only for display) ─────────
NORM_PATCH = np.array([
    [-0.491, -0.542, -0.611, -0.234,  1.050,  1.221,  1.102, -0.371],
    [-0.611, -0.679, -0.577, -0.149,  1.170,  1.478,  1.324, -0.268],
    [-0.714, -0.748, -0.508, -0.063,  1.273,  1.564,  1.461, -0.200],
    [-0.782, -0.816, -0.440, -0.268,  1.136,  1.444,  1.221, -0.320],
    [-0.542, -0.611, -0.371, -0.491,  0.365,  0.879,  0.759, -0.440],
    [-0.440, -0.508, -0.303, -0.577, -0.200,  0.108,  0.023, -0.542],
    [-0.320, -0.405, -0.423, -0.611, -0.491, -0.405, -0.440, -0.611],
    [-0.234, -0.268, -0.491, -0.662, -0.611, -0.542, -0.577, -0.679],
], dtype=np.float32)

CELL = 0.60
MINI_CELL = 0.38

def norm_label(v):
    from utils.styles import norm_color
    col = norm_color(float(v))
    fg  = WHITE if abs(v) < 0.6 else "#111"
    return f"{v:.2f}", col, fg

def make_matrix(data, label_fn, cell_size=CELL):
    rows, cols = data.shape
    if rows == 0 or cols == 0:
        sq = Square(side_length=cell_size)
        sq.set_fill(BORDER, opacity=1)
        return VGroup(VGroup(sq, Text("?", font_size=10)))
    group = VGroup()
    for r in range(rows):
        for c in range(cols):
            val = float(data[r, c])
            disp, bg, fg = label_fn(val)
            sq = Square(side_length=cell_size)
            sq.set_fill(bg, opacity=1).set_stroke(BORDER, width=0.7)
            sq.move_to(RIGHT * c * cell_size + DOWN * r * cell_size)
            txt = Text(disp, font_size=9, color=fg)
            txt.move_to(sq)
            group.add(VGroup(sq, txt))
    return group


def make_feature_map(rows, cols, cell_size, values=None, color=TEAL):
    """Mini feature map grid filled with mock activation values."""
    group = VGroup()
    for r in range(rows):
        for c in range(cols):
            v = values[r, c] if values is not None else np.random.rand()
            sq = Square(side_length=cell_size)
            sq.set_fill(color, opacity=float(np.clip(v, 0.1, 0.95)))
            sq.set_stroke(BG, width=0.5)
            sq.move_to(RIGHT * c * cell_size + DOWN * r * cell_size)
            group.add(sq)
    return group


def make_neuron_row(values, color=WHITE, cell_w=0.42, cell_h=0.42):
    """A row of neurons showing activation values."""
    group = VGroup()
    for i, v in enumerate(values):
        sq = Square(side_length=cell_h)
        sq.set_fill(color, opacity=float(np.clip(abs(v)/2, 0.1, 0.95)))
        sq.set_stroke(BORDER, width=0.8)
        sq.move_to(RIGHT * i * (cell_w + 0.04))
        txt = Text(f"{v:.2f}", font_size=8, color=WHITE)
        txt.move_to(sq)
        group.add(VGroup(sq, txt))
    return group


def safe_arrow(start, end, **kwargs):
    if np.linalg.norm(np.array(end) - np.array(start)) < 0.05:
        end = np.array(start) + np.array([0.5, 0, 0])
    return Arrow(start, end, **kwargs)


def make_probability_bar(label, value, bar_color=BRAND, width=3.5):
    track = Rectangle(width=width, height=0.38)
    track.set_fill(BORDER, opacity=1).set_stroke(BORDER, width=0)
    fill = Rectangle(width=max(0.01, width * value), height=0.38)
    fill.set_fill(bar_color, opacity=1).set_stroke(color=bar_color, width=0)
    fill.align_to(track, LEFT)
    lbl_m = Text(label, font_size=14, color=WHITE)
    lbl_m.next_to(track, LEFT, buff=0.15)
    pct_m = Text(f"{value * 100:.1f}%", font_size=14, color=WHITE)
    pct_m.next_to(track, RIGHT, buff=0.15)
    return VGroup(track, fill, lbl_m, pct_m)


class SceneBTrainingTensor(Scene):
    def construct(self):
        self.camera.background_color = BG
        np.random.seed(42)

        # ── TITLE ──────────────────────────────────────────────────────────────
        title = T("Training — One Sample, Every Step", size=32, color=WHITE)
        sub   = T("8×8 patch from a PNEUMONIA X-ray   |   Label = 1", size=17, color=RED)
        sub.next_to(title, DOWN, buff=0.2)
        VGroup(title, sub).move_to(ORIGIN)
        self.play(FadeIn(title, shift=UP*0.3), FadeIn(sub))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(sub))

        # ══════════════════════════════════════════════════════════════════
        # 1. TENSOR REPRESENTATION (3 channels)
        # ══════════════════════════════════════════════════════════════════
        sec_lbl = T("1 — Input Tensor   shape: (3, 224, 224)", size=22, color=YELLOW)
        sec_lbl.to_edge(UP, buff=0.3)
        self.play(FadeIn(sec_lbl))

        # Show 3 stacked channel matrices (offset like 3D stack)
        offsets = [LEFT*3.2+DOWN*0.4, LEFT*3.0+DOWN*0.2, LEFT*2.8+UP*0.0]
        channel_labels = ["R  (−μ_R)/σ_R", "G  (−μ_G)/σ_G", "B  (−μ_B)/σ_B"]
        ch_colors = [RED, GREEN, BRAND]

        mats = VGroup()
        for i, (off, lbl_str, col) in enumerate(zip(offsets, channel_labels, ch_colors)):
            mat = make_matrix(NORM_PATCH * (1 - i*0.08), norm_label, cell_size=MINI_CELL)
            mat.move_to(off)
            ch_lbl = T(lbl_str, size=13, color=col)
            ch_lbl.next_to(mat, UP, buff=0.1)
            mats.add(VGroup(mat, ch_lbl))

        for m in mats:
            self.play(FadeIn(m, shift=RIGHT*0.15), run_time=0.4)

        # Shape annotation
        shape_box = make_card(4.0, 2.2)
        shape_box.move_to(RIGHT * 4.0)
        shape_lines = VGroup(
            T("Tensor shape", size=16, color=MUTED),
            T("(1, 3, 224, 224)", size=22, color=WHITE),
            T("batch=1", size=14, color=MUTED),
            T("channels=3  (RGB)", size=14, color=MUTED),
            T("height=224", size=14, color=MUTED),
            T("width=224", size=14, color=MUTED),
        )
        shape_lines.arrange(DOWN, buff=0.13).move_to(shape_box)
        self.play(FadeIn(shape_box), FadeIn(shape_lines))
        self.wait(1.5)
        self.play(FadeOut(mats), FadeOut(shape_box), FadeOut(shape_lines), FadeOut(sec_lbl))

        # ══════════════════════════════════════════════════════════════════
        # 2. FORWARD PASS — DenseNet-121 branch
        # ══════════════════════════════════════════════════════════════════
        sec2 = T("2 — DenseNet-121 Forward Pass   (local features)", size=22, color=DENSE_COLOR)
        sec2.to_edge(UP, buff=0.3)
        self.play(FadeIn(sec2))

        # Show the feature map shrinking through DenseBlocks
        stages = [
            ("Input\n(3, 224, 224)",    8, 8,  0.18,  1.0),
            ("DenseBlock1\n(64, 56, 56)", 6, 6, 0.22, 0.65),
            ("DenseBlock2\n(128, 28, 28)", 5, 5, 0.26, 0.75),
            ("DenseBlock3\n(256, 14, 14)", 4, 4, 0.30, 0.85),
            ("DenseBlock4\n(1024, 7, 7)",  3, 3, 0.38, 0.9),
            ("GAP\n(1024,)",             1, 8, 0.38, 0.7),
        ]

        stage_mobs = VGroup()
        x_pos = -5.8
        for label, rows, cols, cs, intensity in stages:
            if rows == 1:
                # Show as a vertical neuron column
                fmap = VGroup()
                for r in range(cols):
                    sq = Square(side_length=cs)
                    sq.set_fill(DENSE_COLOR, opacity=0.3 + 0.6*np.random.rand())
                    sq.set_stroke(BG, width=0.4)
                    sq.move_to(DOWN * r * cs)
                    fmap.add(sq)
            else:
                vals = np.random.rand(rows, cols) * intensity
                fmap = make_feature_map(rows, cols, cs, vals, color=DENSE_COLOR)

            fmap.move_to(RIGHT * x_pos + UP * 0.2)
            lbl  = T(label, size=11, color=DENSE_COLOR)
            lbl.next_to(fmap, DOWN, buff=0.18)
            stage_mobs.add(VGroup(fmap, lbl))
            x_pos += 2.0

        arrows_dense = VGroup()
        for i in range(len(stage_mobs)-1):
            a = safe_arrow(stage_mobs[i][0].get_right() + RIGHT*0.05,
                           stage_mobs[i+1][0].get_left() + LEFT*0.05,
                           color=DENSE_COLOR, buff=0, stroke_width=1.8,
                           tip_length=0.14)
            arrows_dense.add(a)

        for mob, *arr in zip(stage_mobs, [None]+list(arrows_dense)):
            if arr[0]: self.play(GrowArrow(arr[0]), run_time=0.2)
            self.play(FadeIn(mob), run_time=0.35)

        # Annotate: "each block learns to detect edges → textures → opacities"
        annots = [
            (0, "Raw pixels"),
            (1, "Edges"),
            (2, "Textures"),
            (3, "Opacity\nregions"),
            (4, "Complex\npatterns"),
            (5, "1024-d\nvector"),
        ]
        annot_mobs = VGroup()
        for i, (_, txt) in enumerate(annots):
            a = T(txt, size=10, color=MUTED)
            a.next_to(stage_mobs[i], UP, buff=0.12)
            annot_mobs.add(a)
        self.play(FadeIn(annot_mobs))
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)))

        # ══════════════════════════════════════════════════════════════════
        # 3. FORWARD PASS — Swin Transformer branch
        # ══════════════════════════════════════════════════════════════════
        sec3 = T("3 — Swin Transformer Forward Pass   (global context)", size=22, color=SWIN_COLOR)
        sec3.to_edge(UP, buff=0.3)
        self.play(FadeIn(sec3))

        # Show patch partition: 224×224 → 56×56 tokens (4×4 patches)
        patch_grid_outer = Square(side_length=2.8)
        patch_grid_outer.set_fill("#1a1a2e", opacity=1).set_stroke(BORDER, width=1.5)
        patch_grid_outer.move_to(LEFT*4.5)

        # Draw 7×7 patch lines on top
        patch_lines = VGroup()
        step = 2.8/7
        for i in range(1, 7):
            patch_lines.add(
                Line(patch_grid_outer.get_left() + RIGHT*(i*step),
                     patch_grid_outer.get_left() + RIGHT*(i*step) + UP*2.8,
                     color=SWIN_COLOR, stroke_width=0.8)
            )
            patch_lines.add(
                Line(patch_grid_outer.get_bottom() + UP*(i*step),
                     patch_grid_outer.get_bottom() + UP*(i*step) + RIGHT*2.8,
                     color=SWIN_COLOR, stroke_width=0.8)
            )

        in_lbl = T("Input\n224×224", size=14, color=MUTED)
        in_lbl.next_to(patch_grid_outer, DOWN, buff=0.15)
        self.play(FadeIn(patch_grid_outer), FadeIn(in_lbl))
        self.play(Create(patch_lines, run_time=0.8))

        token_lbl = T("Divided into 4×4 patches\n→  3136 patch tokens", size=14, color=SWIN_COLOR)
        token_lbl.next_to(patch_grid_outer, UP, buff=0.2)
        self.play(FadeIn(token_lbl))
        self.wait(0.6)

        # Show window attention: highlight a 7×7 window
        win_size = step * 7  # the entire grid
        win_box = Square(side_length=step * 3)
        win_box.set_stroke(YELLOW, width=2.5).set_fill(YELLOW, opacity=0.12)
        win_box.move_to(patch_grid_outer.get_center() + UP*0.3 + RIGHT*0.2)
        win_lbl = T("Window\n(7×7 tokens)", size=12, color=YELLOW)
        win_lbl.next_to(win_box, RIGHT, buff=0.1)
        self.play(FadeIn(win_box), FadeIn(win_lbl))
        self.wait(0.6)

        # Shift: move the window
        self.play(win_box.animate.shift(DOWN*0.42+RIGHT*0.42), run_time=0.7)
        shift_lbl = T("Shifted Window\n(captures cross-patch context)", size=12, color=ORANGE)
        shift_lbl.next_to(win_box, RIGHT, buff=0.1)
        self.play(FadeIn(shift_lbl))
        self.wait(0.8)

        # Show Swin stages → output token
        swin_stages = [
            ("Stage 1\n(C=96, 56×56)",  5, 5, 0.22, 0.5),
            ("Stage 2\n(C=192, 28×28)", 4, 4, 0.28, 0.6),
            ("Stage 3\n(C=384, 14×14)", 3, 3, 0.34, 0.7),
            ("Stage 4\n(C=768, 7×7)",   2, 2, 0.44, 0.85),
            ("Flatten\n(768,)",          1, 6, 0.38, 0.7),
        ]

        swin_mobs = VGroup()
        x_s = -1.0
        for label, rows, cols, cs, intensity in swin_stages:
            if rows == 1:
                fmap = VGroup()
                for r in range(cols):
                    sq = Square(side_length=cs)
                    sq.set_fill(SWIN_COLOR, opacity=0.3 + 0.6*np.random.rand())
                    sq.set_stroke(BG, width=0.4)
                    sq.move_to(RIGHT * r * cs)
                    fmap.add(sq)
            else:
                vals = np.random.rand(rows, cols) * intensity
                fmap = make_feature_map(rows, cols, cs, vals, color=SWIN_COLOR)
            fmap.move_to(RIGHT * x_s + DOWN*0.0)
            lbl = T(label, size=11, color=SWIN_COLOR)
            lbl.next_to(fmap, DOWN, buff=0.16)
            swin_mobs.add(VGroup(fmap, lbl))
            x_s += 1.85

        arrows_swin = VGroup()
        for i in range(len(swin_mobs)-1):
            a = safe_arrow(swin_mobs[i][0].get_right()+RIGHT*0.05,
                           swin_mobs[i+1][0].get_left()+LEFT*0.05,
                           color=SWIN_COLOR, buff=0, stroke_width=1.8, tip_length=0.14)
            arrows_swin.add(a)

        for mob, *arr in zip(swin_mobs, [None]+list(arrows_swin)):
            if arr[0]: self.play(GrowArrow(arr[0]), run_time=0.2)
            self.play(FadeIn(mob), run_time=0.35)

        self.wait(1.2)
        self.play(FadeOut(Group(*self.mobjects)))

        # ══════════════════════════════════════════════════════════════════
        # 4. CONCATENATION + MLP HEAD (neuron-level detail)
        # ══════════════════════════════════════════════════════════════════
        sec4 = T("4 — Fusion: Concat → MLP Head", size=22, color=FUSE_COLOR)
        sec4.to_edge(UP, buff=0.3)
        self.play(FadeIn(sec4))

        # Show DenseNet vector (show 12 neurons)
        d_vals  = np.clip(np.random.randn(12)*0.6 + 0.3, -1.5, 1.5).round(2)
        s_vals  = np.clip(np.random.randn(12)*0.5 + 0.2, -1.5, 1.5).round(2)

        d_row   = make_neuron_row(d_vals, color=DENSE_COLOR)
        d_row.move_to(LEFT*2.0 + UP*2.2)
        d_lbl   = T("DenseNet  (1024-d shown: 12)", size=13, color=DENSE_COLOR)
        d_lbl.next_to(d_row, LEFT, buff=0.2)

        s_row   = make_neuron_row(s_vals, color=SWIN_COLOR)
        s_row.move_to(LEFT*2.0 + UP*1.4)
        s_lbl   = T("Swin Transformer  (768-d shown: 12)", size=13, color=SWIN_COLOR)
        s_lbl.next_to(s_row, LEFT, buff=0.2)

        self.play(FadeIn(d_row), FadeIn(d_lbl))
        self.play(FadeIn(s_row), FadeIn(s_lbl))

        # Concat arrow
        concat_lbl = T("torch.cat([f_dense, f_swin], dim=1)\n→  (1, 1792)", size=15, color=BRAND_LIGHT)
        concat_lbl.move_to(UP * 0.5)
        cat_arrow = safe_arrow(
            VGroup(d_row, s_row).get_bottom() + DOWN*0.1,
            concat_lbl.get_top() + UP*0.05,
            color=BRAND_LIGHT, buff=0
        )
        self.play(GrowArrow(cat_arrow), Write(concat_lbl))
        self.wait(0.5)

        # Show MLP layers with actual neuron counts
        mlp_layers = [
            ("Linear(1792→512)\n+ BN + ReLU + Drop(0.4)", 512,  BRAND, 10),
            ("Linear(512→128)\n+ BN + ReLU + Drop(0.2)",  128,  BRAND, 8),
            ("Linear(128→2)",                              2,    GREEN, 2),
        ]

        prev_group = concat_lbl
        layer_mobs = VGroup()
        for layer_lbl, out_dim, col, show_n in mlp_layers:
            # Show 'show_n' sample neurons
            sample_vals = np.clip(np.random.randn(show_n)*0.7, -1.5, 1.5).round(2)
            n_row = make_neuron_row(sample_vals, color=col, cell_w=0.38, cell_h=0.38)

            # If only 2 neurons, show actual logit values for NORMAL / PNEUMONIA
            if out_dim == 2:
                logit_normal   = -2.18
                logit_pneumonia = 3.74
                sample_vals = np.array([logit_normal, logit_pneumonia])
                n_row = make_neuron_row(sample_vals, color=GREEN, cell_w=0.55, cell_h=0.55)

            n_row.next_to(prev_group, DOWN, buff=0.35)
            ll = T(layer_lbl, size=12, color=col)
            ll.next_to(n_row, RIGHT, buff=0.3)
            la = safe_arrow(prev_group.get_bottom() + DOWN*0.05,
                            n_row.get_top() + UP*0.05,
                            color=col, buff=0, stroke_width=1.5, tip_length=0.12)
            self.play(GrowArrow(la), FadeIn(n_row), FadeIn(ll), run_time=0.5)
            layer_mobs.add(VGroup(n_row, ll))
            prev_group = n_row

        # Show softmax output
        softmax_lbl = T("Softmax", size=14, color=MUTED)
        softmax_lbl.next_to(prev_group, DOWN, buff=0.3)
        self.play(FadeIn(softmax_lbl))

        prob_normal = 0.018
        prob_pneumonia = 0.982
        bar_n = make_probability_bar("NORMAL", prob_normal, TEAL)
        bar_p = make_probability_bar("PNEUMONIA", prob_pneumonia, RED)
        bar_n.next_to(softmax_lbl, DOWN, buff=0.2)
        bar_p.next_to(bar_n, DOWN, buff=0.2)
        self.play(FadeIn(bar_n), FadeIn(bar_p))

        pred_box = make_card(3.5, 0.7, fill=PANEL, stroke=RED)
        pred_txt = T("→  PNEUMONIA  (p = 0.982)", size=16, color=RED)
        pred_box.next_to(bar_p, DOWN, buff=0.25)
        pred_txt.move_to(pred_box)
        self.play(FadeIn(pred_box), FadeIn(pred_txt))
        self.wait(1.2)
        self.play(FadeOut(Group(*self.mobjects)))

        # ══════════════════════════════════════════════════════════════════
        # 5. LOSS CALCULATION (step by step with real numbers)
        # ══════════════════════════════════════════════════════════════════
        sec5 = T("5 — Loss Calculation   (Weighted CrossEntropy + Label Smoothing)", size=20, color=YELLOW)
        sec5.to_edge(UP, buff=0.3)
        self.play(FadeIn(sec5))

        # True label
        true_lbl   = T("True label:  y = 1  (PNEUMONIA)", size=18, color=RED)
        true_lbl.move_to(UP*2.5)
        self.play(FadeIn(true_lbl))

        # Logits
        logit_txt = T("Logits:  [ −2.18,   3.74 ]", size=18, color=WHITE)
        logit_txt.next_to(true_lbl, DOWN, buff=0.3)
        self.play(FadeIn(logit_txt))

        # Softmax step by step
        import math
        e0 = math.exp(-2.18)
        e1 = math.exp(3.74)
        s  = e0 + e1
        p0 = e0/s
        p1 = e1/s

        softmax_steps = VGroup(
            T(f"exp(−2.18) = {e0:.4f}   exp(3.74) = {e1:.4f}", size=16, color=BRAND_LIGHT),
            T(f"sum = {e0:.4f} + {e1:.4f} = {s:.4f}", size=16, color=BRAND_LIGHT),
            T(f"P(NORMAL)    = {e0:.4f} / {s:.4f} = {p0:.4f}", size=16, color=TEAL),
            T(f"P(PNEUMONIA) = {e1:.4f} / {s:.4f} = {p1:.4f}", size=16, color=RED),
        )
        softmax_steps.arrange(DOWN, buff=0.2).next_to(logit_txt, DOWN, buff=0.35)
        for step in softmax_steps:
            self.play(FadeIn(step, shift=RIGHT*0.15), run_time=0.4)
        self.wait(0.6)

        # Label smoothing
        eps = 0.1
        y_smooth = 1 - eps + eps/2   # = 0.95 for the true class
        smooth_lbl = T(
            f"Label smoothing (ε=0.1):  y_smooth = 1 − 0.1 + 0.1/2 = {y_smooth:.2f}",
            size=16, color=YELLOW
        )
        smooth_lbl.next_to(softmax_steps, DOWN, buff=0.3)
        self.play(FadeIn(smooth_lbl))
        self.wait(0.5)

        # Class weight
        w_pneumonia = 0.6730
        w_lbl = T(f"Class weight for PNEUMONIA:  w = {w_pneumonia}", size=16, color=ORANGE)
        w_lbl.next_to(smooth_lbl, DOWN, buff=0.2)
        self.play(FadeIn(w_lbl))

        # Loss value
        loss_val = -w_pneumonia * y_smooth * math.log(p1)
        loss_lbl = T(
            f"ℒ = − {w_pneumonia} × {y_smooth:.2f} × log({p1:.4f})\n"
            f"ℒ = {loss_val:.4f}",
            size=18, color=WHITE
        )
        loss_lbl.next_to(w_lbl, DOWN, buff=0.3)
        loss_box = make_card(6.5, 1.1, fill=PANEL, stroke=YELLOW)
        loss_box.move_to(loss_lbl.get_center())
        self.play(FadeIn(loss_box), Write(loss_lbl))
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)))

        # ══════════════════════════════════════════════════════════════════
        # 6. BACKWARD PASS — Gradient visualisation
        # ══════════════════════════════════════════════════════════════════
        sec6 = T("6 — Backward Pass   (loss.backward())", size=22, color=ORANGE)
        sec6.to_edge(UP, buff=0.3)
        self.play(FadeIn(sec6))

        backprop_txt = T(
            "Chain rule propagates ∂ℒ/∂W through every layer\n"
            "from the output logits back to the image pixels.",
            size=17, color=WHITE
        )
        backprop_txt.move_to(UP*2.5)
        self.play(FadeIn(backprop_txt))

        # Show the gradient matrix for the input patch
        # Gradients are largest where the model's evidence is: the opacity region
        grad_patch = np.array([
            [-0.02, -0.01, -0.01, -0.03,  0.18,  0.22,  0.19, -0.02],
            [-0.01, -0.01, -0.01, -0.04,  0.20,  0.25,  0.22, -0.02],
            [-0.01, -0.01, -0.01, -0.04,  0.21,  0.27,  0.23, -0.01],
            [-0.01, -0.01, -0.02, -0.03,  0.19,  0.24,  0.20, -0.02],
            [-0.01, -0.01, -0.01, -0.02,  0.08,  0.13,  0.11, -0.02],
            [-0.01, -0.01, -0.01, -0.02, -0.01,  0.02,  0.01, -0.01],
            [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
            [-0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01, -0.01],
        ], dtype=np.float32)

        def grad_label(v):
            t = (v + 0.3) / 0.6
            t = float(np.clip(t, 0, 1))
            if t < 0.5:
                col = "#{:02x}{:02x}ff".format(int(30*(1-2*t)), int(100*(1-2*t)))
            else:
                col = "#ff{:02x}{:02x}".format(int(30*(2*t-1)), int(30*(2*t-1)))
            fg = WHITE
            return f"{v:+.2f}", col, fg

        grad_mat = make_matrix(grad_patch, grad_label, cell_size=CELL)
        grad_mat.move_to(LEFT*2.0 + DOWN*0.3)

        grad_title = T("∂ℒ/∂x  (gradient w.r.t. input patch)", size=16, color=ORANGE)
        grad_title.next_to(grad_mat, UP, buff=0.2)

        self.play(FadeIn(grad_mat), FadeIn(grad_title))

        # Annotate: high gradient = model's attention
        high_grad = T(
            "Large gradients on the\nopacity region (cols 4-6)\n→ model relies on this area",
            size=14, color=RED
        )
        high_grad.move_to(RIGHT*4.0 + UP*0.5)
        arr_hg = safe_arrow(high_grad.get_left(), grad_mat[2*8+5][0].get_right(),
                            color=RED, buff=0.1, stroke_width=1.5, tip_length=0.14)
        self.play(FadeIn(high_grad), GrowArrow(arr_hg))
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)))

        # ══════════════════════════════════════════════════════════════════
        # 7. WEIGHT UPDATE — AdamW step
        # ══════════════════════════════════════════════════════════════════
        sec7 = T("7 — Weight Update   (AdamW optimizer)", size=22, color=GREEN)
        sec7.to_edge(UP, buff=0.3)
        self.play(FadeIn(sec7))

        # Show ONE weight changing
        w_before = 0.3214
        grad_w    = 0.0089
        lr_head   = 1e-3
        w_after   = round(w_before - lr_head * grad_w, 6)

        update_steps = VGroup(
            T(f"W  (before):       {w_before}", size=18, color=WHITE),
            T(f"∂ℒ/∂W:             {grad_w}", size=18, color=ORANGE),
            T(f"LR (head):          {lr_head}", size=18, color=BRAND_LIGHT),
            T(f"AdamW step ≈  W − LR × grad", size=18, color=MUTED),
            T(f"W  (after):        {w_after}", size=18, color=GREEN),
        )
        update_steps.arrange(DOWN, buff=0.28).move_to(LEFT*1.5 + UP*0.5)

        for step in update_steps:
            self.play(FadeIn(step, shift=RIGHT*0.2), run_time=0.4)
        self.wait(0.5)

        # Show gradient clipping
        clip_box = make_card(5.5, 1.0, fill=PANEL, stroke=YELLOW)
        clip_box.move_to(RIGHT*3.8 + UP*1.5)
        clip_txt = T("Gradient clip:\nmax_norm = 1.0  (prevents explosion in Swin)", size=14, color=YELLOW)
        clip_txt.move_to(clip_box)
        self.play(FadeIn(clip_box), FadeIn(clip_txt))

        # Show differential LR
        dlr_box = make_card(5.5, 1.2, fill=PANEL, stroke=TEAL)
        dlr_box.next_to(clip_box, DOWN, buff=0.3)
        dlr_txt = T(
            "Differential LR:\n"
            "Backbone (Swin+DenseNet): 1e-5\n"
            "Fusion head:              1e-3",
            size=14, color=TEAL
        )
        dlr_txt.move_to(dlr_box)
        self.play(FadeIn(dlr_box), FadeIn(dlr_txt))

        # Scheduler
        sched_box = make_card(5.5, 0.9, fill=PANEL, stroke=PURPLE)
        sched_box.next_to(dlr_box, DOWN, buff=0.3)
        sched_txt = T("Scheduler: CosineAnnealingWarmRestarts(T₀=10)", size=14, color=PURPLE)
        sched_txt.move_to(sched_box)
        self.play(FadeIn(sched_box), FadeIn(sched_txt))
        self.wait(2.0)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.3)
