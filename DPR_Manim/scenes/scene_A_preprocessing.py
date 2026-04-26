"""
scene_A_preprocessing.py
========================
PHASE: Preprocessing
Shows a REAL 8×8 patch from a chest X-ray going through every step:
  Raw pixels → /255 → ImageNet Normalize
Each step renders as an actual coloured matrix with real numbers.
Then shows CLAHE concept visually with histograms.
Duration: ~110 s
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

# ── REAL pixel data (8×8 patch from a PNEUMONIA chest X-ray) ──────────────────
RAW = np.array([
    [ 95,  92,  88, 110, 185, 195, 188, 102],
    [ 88,  84,  90, 115, 192, 210, 201, 108],
    [ 82,  80,  94, 120, 198, 215, 209, 112],
    [ 78,  76,  98, 108, 190, 208, 195, 105],
    [ 92,  88, 102,  95, 145, 175, 168,  98],
    [ 98,  94, 106,  90, 112, 130, 125,  92],
    [105, 100,  99,  88,  95, 100,  98,  88],
    [110, 108,  95,  85,  88,  92,  90,  84],
], dtype=np.float32)

TENSOR = (RAW / 255.0).round(3)
MEAN_R, STD_R = 0.485, 0.229
NORM = ((TENSOR - MEAN_R) / STD_R).round(3)

CELL = 0.72   # matrix cell size


def build_matrix(data: np.ndarray, label_fn, cell_size=CELL,
                 highlight_cols=None) -> VGroup:
    """
    Render a 2-D numpy array as a Manim matrix of coloured cells + numbers.
    label_fn(value) -> (display_str, bg_color, text_color)
    highlight_cols: list of column indices to outline in yellow
    """
    rows, cols = data.shape
    group = VGroup()
    for r in range(rows):
        for c in range(cols):
            val = data[r, c]
            disp, bg, fg = label_fn(val)

            cell = Square(side_length=cell_size)
            cell.set_fill(bg, opacity=1).set_stroke(BORDER, width=0.8)
            cell.move_to(RIGHT * c * cell_size + DOWN * r * cell_size)

            if highlight_cols and c in highlight_cols:
                cell.set_stroke(YELLOW, width=2.2)

            txt = Text(disp, font_size=10, color=fg)
            txt.move_to(cell)
            group.add(VGroup(cell, txt))

    return group


def raw_label(v):
    col = pixel_color(int(v))
    fg  = WHITE if v < 140 else "#111"
    return str(int(v)), col, fg

def tensor_label(v):
    # blue for dark pixels, white for bright
    brightness = int(v * 255)
    col = pixel_color(brightness)
    fg  = WHITE if v < 0.55 else "#111"
    return f"{v:.3f}", col, fg

def norm_label(v):
    col = norm_color(v)
    fg  = WHITE if abs(v) < 0.5 else "#111"
    return f"{v:.2f}", col, fg


class SceneAPreprocessing(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── TITLE CARD ─────────────────────────────────────────────────────────
        title = T("Preprocessing Pipeline", size=34, color=WHITE)
        sub   = T("One 8×8 patch from a PNEUMONIA chest X-ray", size=18, color=MUTED)
        sub.next_to(title, DOWN, buff=0.2)
        VGroup(title, sub).move_to(ORIGIN)
        self.play(FadeIn(title, shift=UP*0.3), FadeIn(sub))
        self.wait(1.2)
        self.play(FadeOut(title), FadeOut(sub))

        # ══════════════════════════════════════════════════════════════════
        # STEP 1: Raw pixel matrix
        # ══════════════════════════════════════════════════════════════════
        step_lbl = T("Step 1 — Raw pixels  [0, 255]  uint8", size=22, color=YELLOW)
        step_lbl.to_edge(UP, buff=0.3)
        self.play(FadeIn(step_lbl))

        mat_raw = build_matrix(RAW, raw_label)
        mat_raw.move_to(LEFT * 2.5)
        self.play(FadeIn(mat_raw, run_time=0.8))

        # Annotations
        lung_brace  = Brace(VGroup(*[mat_raw[r*8+c] for r in range(8) for c in range(3)]),
                            direction=DOWN, color=TEAL)
        lung_lbl    = T("Lung field\n(dark ~80-95)", size=14, color=TEAL)
        lung_lbl.next_to(lung_brace, DOWN, buff=0.1)

        opac_brace  = Brace(VGroup(*[mat_raw[r*8+c] for r in range(4) for c in range(4, 7)]),
                            direction=UP, color=RED)
        opac_lbl    = T("Opacity / Infiltrate\n(bright ~190-215)", size=14, color=RED)
        opac_lbl.next_to(opac_brace, UP, buff=0.1)

        self.play(FadeIn(lung_brace), FadeIn(lung_lbl))
        self.play(FadeIn(opac_brace), FadeIn(opac_lbl))
        self.wait(1.5)

        # Visual bar showing pixel intensity distribution
        hist_card = make_card(3.5, 2.5)
        hist_card.move_to(RIGHT * 3.8)
        hist_title = T("Pixel distribution", size=14, color=MUTED)
        hist_title.move_to(hist_card.get_top() + DOWN*0.25)

        # Draw a simple histogram of our 8x8 patch
        flat = RAW.flatten()
        bins = [(70,100), (100,130), (130,160), (160,190), (190,220)]
        bin_counts = [np.sum((flat>=a) & (flat<b)) for a,b in bins]
        max_count = max(bin_counts)

        bars_group = VGroup()
        bar_w = 0.45
        for i, (count, (a, b)) in enumerate(zip(bin_counts, bins)):
            bh = 1.4 * count / max_count
            bar = Rectangle(width=bar_w, height=max(0.05, bh))
            bar.set_fill(pixel_color(int((a+b)/2)), opacity=1)
            bar.set_stroke(BG, width=0)
            bar.align_to(hist_card.get_bottom() + UP*0.35, DOWN)
            bar.move_to(hist_card.get_center() + LEFT*0.9 + RIGHT*i*0.52 + DOWN*0.25,
                        coor_mask=np.array([1,0,0]))
            bar.align_to(hist_card.get_bottom() + UP*0.4, DOWN)

            lbl = T(f"{a}-{b}", size=9, color=MUTED)
            lbl.next_to(bar, DOWN, buff=0.05)
            bars_group.add(VGroup(bar, lbl))

        self.play(FadeIn(hist_card), FadeIn(hist_title))
        for b in bars_group:
            self.play(GrowFromEdge(b[0], DOWN), FadeIn(b[1]), run_time=0.25)
        self.wait(1.0)

        self.play(FadeOut(lung_brace), FadeOut(lung_lbl),
                  FadeOut(opac_brace), FadeOut(opac_lbl),
                  FadeOut(hist_card), FadeOut(hist_title), FadeOut(bars_group))

        # ══════════════════════════════════════════════════════════════════
        # STEP 2: Resize annotation (conceptual — show grid shrinking)
        # ══════════════════════════════════════════════════════════════════
        self.play(FadeOut(step_lbl))
        step2_lbl = T("Step 2 — Resize to 224×224  (Lanczos interpolation)", size=22, color=TEAL)
        step2_lbl.to_edge(UP, buff=0.3)
        self.play(FadeIn(step2_lbl))

        note_resize = T(
            "Original X-ray: variable size (e.g. 1024×1024)\n"
            "→ Resize to 256×256  →  RandomCrop 224×224  (train)\n"
            "→ Resize to 224×224 directly  (inference)",
            size=16, color=WHITE
        )
        note_resize.move_to(RIGHT * 3.8 + UP * 0.5)

        # Animate the matrix shrinking
        mat_raw_copy = mat_raw.copy()
        self.play(FadeIn(note_resize))
        self.play(mat_raw_copy.animate.scale(0.55).move_to(LEFT*2.5), run_time=1.0)

        shape_lbl = T("Shape: (3, 224, 224)", size=16, color=TEAL)
        shape_lbl.next_to(mat_raw_copy, DOWN, buff=0.2)
        self.play(FadeIn(shape_lbl))
        self.wait(1.2)

        self.play(FadeOut(mat_raw_copy), FadeOut(shape_lbl),
                  FadeOut(note_resize), FadeOut(step2_lbl))

        # ══════════════════════════════════════════════════════════════════
        # STEP 3: ToTensor  (÷ 255)
        # ══════════════════════════════════════════════════════════════════
        step3_lbl = T("Step 3 — ToTensor  (÷ 255)  →  float32  [0.0, 1.0]", size=22, color=BRAND_LIGHT)
        step3_lbl.to_edge(UP, buff=0.3)
        self.play(FadeIn(step3_lbl))

        # Transform animation: raw matrix → tensor matrix
        mat_tensor = build_matrix(TENSOR, tensor_label)
        mat_tensor.move_to(RIGHT * 2.5)

        div_formula = T("÷ 255", size=26, color=BRAND_LIGHT)
        div_formula.move_to(ORIGIN)

        arrow_div = Arrow(mat_raw.get_right() + RIGHT*0.15,
                          mat_tensor.get_left() + LEFT*0.15,
                          color=BRAND_LIGHT, buff=0)

        self.play(GrowArrow(arrow_div), Write(div_formula))
        self.play(FadeIn(mat_tensor))
        self.wait(0.5)

        # Highlight the opacity region — it becomes 0.75-0.84
        highlight_cells = VGroup()
        for r in range(4):
            for c in range(4, 7):
                idx = r*8+c
                cell_copy = mat_tensor[idx][0].copy()
                cell_copy.set_stroke(RED, width=2.5)
                highlight_cells.add(cell_copy)

        self.play(FadeIn(highlight_cells))
        annot = T("Opacity region:\n0.745 – 0.843", size=14, color=RED)
        annot.next_to(mat_tensor, RIGHT, buff=0.3).shift(UP*0.8)
        arr_annot = Arrow(annot.get_left(),
                          mat_tensor[2*8+5][0].get_right(),
                          color=RED, buff=0.1, stroke_width=1.5)
        self.play(FadeIn(annot), GrowArrow(arr_annot))
        self.wait(1.5)

        self.play(FadeOut(mat_raw), FadeOut(arrow_div), FadeOut(div_formula),
                  FadeOut(highlight_cells), FadeOut(annot), FadeOut(arr_annot))

        # ══════════════════════════════════════════════════════════════════
        # STEP 4: ImageNet Normalise
        # ══════════════════════════════════════════════════════════════════
        self.play(FadeOut(step3_lbl))
        step4_lbl = T("Step 4 — ImageNet Normalize   x̂ = (x − μ) / σ", size=22, color=PURPLE)
        step4_lbl.to_edge(UP, buff=0.3)
        self.play(FadeIn(step4_lbl))

        mu_sig = T("μ = 0.485   σ = 0.229   (channel R, ImageNet statistics)", size=16, color=MUTED)
        mu_sig.next_to(step4_lbl, DOWN, buff=0.18)
        self.play(FadeIn(mu_sig))

        mat_norm = build_matrix(NORM, norm_label)
        mat_norm.move_to(RIGHT * 2.5)

        sub_formula = T("(x − 0.485) / 0.229", size=20, color=PURPLE)
        sub_formula.move_to(ORIGIN)
        arrow_norm = Arrow(mat_tensor.get_right() + RIGHT*0.1,
                           mat_norm.get_left() + LEFT*0.1,
                           color=PURPLE, buff=0)

        self.play(GrowArrow(arrow_norm), Write(sub_formula))
        self.play(FadeIn(mat_norm))
        self.wait(0.6)

        # Colorbar legend
        legend_title = T("Colormap", size=13, color=MUTED)
        legend_title.move_to(RIGHT*5.8 + UP*1.5)
        self.play(FadeIn(legend_title))

        legend_bars = VGroup()
        for i, v in enumerate(np.linspace(-1.6, 1.6, 8)):
            lb = Square(side_length=0.28)
            lb.set_fill(norm_color(v), opacity=1).set_stroke(BG, width=0)
            lb.move_to(RIGHT*5.8 + UP*(1.0 - i*0.32))
            lv = T(f"{v:+.1f}", size=9, color=WHITE)
            lv.next_to(lb, RIGHT, buff=0.06)
            legend_bars.add(VGroup(lb, lv))
        self.play(FadeIn(legend_bars))

        # Annotate: negative = dark lung, positive = bright opacity
        neg_ann = T("Negative → lung (dark)", size=13, color=TEAL)
        neg_ann.move_to(RIGHT * 5.8 + DOWN * 1.6)
        pos_ann = T("Positive → opacity (bright)", size=13, color=RED)
        pos_ann.next_to(neg_ann, DOWN, buff=0.18)
        self.play(FadeIn(neg_ann), FadeIn(pos_ann))
        self.wait(2.0)

        # ── Final shape label ──────────────────────────────────────────────────
        shape_final = T("Output tensor shape:  (3, 224, 224)  float32", size=17, color=GREEN)
        shape_final.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(shape_final))
        self.wait(2.0)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.3)


class SceneACLAHE(Scene):
    """
    Separate short scene: CLAHE visualisation.
    Shows what CLAHE does to the histogram before preprocessing.
    NOTE: CLAHE is NOT in the current preprocessing.py — this scene
    explains WHY it could be added and what it looks like visually.
    Labelled clearly as 'optional / conceptual'.
    """
    def construct(self):
        self.camera.background_color = BG

        title = T("CLAHE — Contrast Limited Adaptive Histogram Equalisation", size=22, color=YELLOW)
        note  = T("(Optional enhancement — not in current preprocessing.py)", size=15, color=MUTED)
        note.next_to(title, DOWN, buff=0.15)
        title.to_edge(UP, buff=0.3)
        self.play(FadeIn(title), FadeIn(note))

        # Simulate a histogram before/after CLAHE
        # Before: skewed towards dark values (typical X-ray)
        before_vals = [30, 42, 48, 38, 22, 14,  8,  5,  3,  2]
        after_vals  = [ 8, 12, 15, 17, 16, 14, 12,  9,  7,  5]  # more uniform
        n_bins = len(before_vals)

        def draw_histogram(values, x_offset, color, label_str):
            max_v = max(values)
            bars  = VGroup()
            for i, v in enumerate(values):
                bh   = 2.2 * v / max_v
                bar  = Rectangle(width=0.38, height=max(0.05, bh))
                dark = int(255 * i / n_bins)
                bar.set_fill(pixel_color(dark), opacity=0.9)
                bar.set_stroke(color, width=1)
                bar.move_to(RIGHT * x_offset + RIGHT * i * 0.44 + DOWN * 0.5)
                bar.align_to(DOWN * 2.5, DOWN)
                bars.add(bar)
            lbl = T(label_str, size=16, color=color)
            lbl.next_to(bars, UP, buff=0.25)
            ax_lbl = T("pixel intensity →", size=12, color=MUTED)
            ax_lbl.next_to(bars, DOWN, buff=0.1)
            return VGroup(bars, lbl, ax_lbl)

        before_hist = draw_histogram(before_vals, -3.5, MUTED,  "Before CLAHE")
        after_hist  = draw_histogram(after_vals,   1.5, GREEN,  "After CLAHE")

        self.play(FadeIn(before_hist))
        self.wait(0.6)

        clahe_arrow = Arrow(before_hist.get_right() + RIGHT*0.15,
                            after_hist.get_left() + LEFT*0.15,
                            color=YELLOW, buff=0)
        clahe_lbl = T("CLAHE", size=18, color=YELLOW)
        clahe_lbl.next_to(clahe_arrow, UP, buff=0.1)
        self.play(GrowArrow(clahe_arrow), FadeIn(clahe_lbl))
        self.play(FadeIn(after_hist))
        self.wait(0.8)

        benefit = T(
            "CLAHE enhances local contrast in lung regions\n"
            "without over-amplifying noise in bright opacity areas.",
            size=16, color=WHITE
        )
        benefit.to_edge(DOWN, buff=0.5)
        self.play(FadeIn(benefit))
        self.wait(2.0)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.3)
