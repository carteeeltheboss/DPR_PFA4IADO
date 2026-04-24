"""
scenes/scene_07_evaluation.py
Scene 7 — Evaluation Metrics & Final Pipeline Recap
Covers: confusion matrix, AUC-ROC=0.98, classification report,
        final end-to-end pipeline summary.
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


class Scene07Evaluation(Scene):
    """Test-set results and complete pipeline recap."""

    def construct(self):
        self.camera.background_color = BG

        # ── Title ──────────────────────────────────────────────────────────────
        title = make_section_title("Phase 7 — Evaluation on Test Set")
        self.play(FadeIn(title))
        self.wait(0.9)
        self.play(FadeOut(title))

        # ── 1. Confusion matrix (real results from our test run) ───────────────
        cm_data = [
            [201, 33],
            [13,  377],
        ]
        total = sum(sum(row) for row in cm_data)
        class_names_cm = ["NORMAL", "PNEUMONIA"]

        cm_title = Text("Confusion Matrix  (N=624 test images)",
                        font="Monospace", font_size=20, color=WHITE)
        cm_title.move_to(UP * 3.2)
        self.play(Write(cm_title))

        cell_size = 1.4
        cm_group  = VGroup()

        header_colors = {"NORMAL": TEAL, "PNEUMONIA": RED}

        for r in range(2):
            for c in range(2):
                val   = cm_data[r][c]
                is_tp = (r == c)
                fill  = TEAL if (r == 0 and c == 0) else \
                        RED  if (r == 1 and c == 1) else PANEL
                opacity = 0.35 if is_tp else 0.18

                cell = Square(side_length=cell_size)
                cell.set_fill(fill, opacity=opacity)
                cell.set_stroke(BORDER, width=1.5)
                cell.move_to(RIGHT * c * (cell_size + 0.1) +
                             DOWN  * r * (cell_size + 0.1) + UP * 0.6)

                val_txt = Text(str(val), font="Monospace",
                               font_size=30, color=WHITE, weight=BOLD)
                val_txt.move_to(cell)

                pct = val / total * 100
                pct_txt = small_text(f"{pct:.1f}%", color=MUTED)
                pct_txt.next_to(val_txt, DOWN, buff=0.08)

                cm_group.add(VGroup(cell, val_txt, pct_txt))

        # Centre the matrix
        cm_group.move_to(LEFT * 2.8 + UP * 0.5)

        # Row/col headers
        for i, name in enumerate(class_names_cm):
            col_h = Text(name, font="Monospace", font_size=14,
                         color=list(header_colors.values())[i])
            col_h.next_to(cm_group[i * 2], UP, buff=0.22)
            row_h = Text(name, font="Monospace", font_size=14,
                         color=list(header_colors.values())[i])
            row_h.next_to(cm_group[i * 2][0], LEFT, buff=0.22)
            cm_group.add(col_h, row_h)

        pred_lbl = small_text("Predicted →")
        true_lbl = small_text("← True")
        pred_lbl.next_to(cm_group, UP, buff=0.5)
        true_lbl.next_to(cm_group, LEFT, buff=0.4)

        for mob in [pred_lbl, true_lbl] + list(cm_group):
            self.play(FadeIn(mob, scale=0.9), run_time=0.15)
        self.wait(0.8)

        # ── 2. Key metrics panel ───────────────────────────────────────────────
        metrics = [
            ("AUC-ROC",    "0.98",  GREEN),
            ("Accuracy",   "93.3%", BRAND_LIGHT),
            ("Precision",  "92%",   TEAL),
            ("Recall",     "97%",   ORANGE),
            ("F1-Score",   "0.94",  PURPLE),
            ("Spec.",      "86%",   YELLOW),
        ]

        metric_cards = VGroup()
        for i, (name, val, color) in enumerate(metrics):
            card = make_card(2.1, 1.3, fill=PANEL)
            card.move_to(RIGHT * 2.5 + RIGHT * (i % 3) * 2.3 +
                         UP * (0.9 - (i // 3) * 1.55))
            name_txt = Text(name, font="Monospace", font_size=14, color=MUTED)
            val_txt  = Text(val,  font="Monospace", font_size=28, color=color, weight=BOLD)
            name_txt.move_to(card.get_center() + UP * 0.28)
            val_txt.move_to(card.get_center() + DOWN * 0.18)
            metric_cards.add(VGroup(card, name_txt, val_txt))

        for mc in metric_cards:
            self.play(FadeIn(mc, scale=0.9), run_time=0.22)
        self.wait(1.0)

        # ── 3. Error analysis ──────────────────────────────────────────────────
        fn_note = Text(
            "❌ 13 false negatives  (missed pneumonia — clinically most critical)",
            font="Monospace", font_size=16, color=RED,
        )
        fp_note = Text(
            "⚠  33 false positives  (normal flagged — leads to extra follow-up only)",
            font="Monospace", font_size=16, color=YELLOW,
        )
        fn_note.move_to(DOWN * 3.0)
        fp_note.next_to(fn_note, DOWN, buff=0.22)

        self.play(FadeIn(fn_note))
        self.play(FadeIn(fp_note))
        self.wait(1.2)

        self.play(FadeOut(Group(*self.mobjects)))

        # ══════════════════════════════════════════════════════════════════════
        # FINAL PIPELINE RECAP
        # ══════════════════════════════════════════════════════════════════════
        recap_title = make_section_title("Full Pipeline — End to End")
        self.play(FadeIn(recap_title))
        self.wait(0.7)
        self.play(recap_title.animate.to_edge(UP, buff=0.3).scale(0.8))

        pipeline = [
            ("X-ray\n(DICOM / JPEG)",         WHITE,        "Input"),
            ("Resize 224²\nNormalize",         TEAL,         "Preprocess"),
            ("DenseNet-121\n(local features)", DENSE_COLOR,  "Branch A"),
            ("Swin Transformer\n(global ctx)", SWIN_COLOR,   "Branch B"),
            ("Concat 1792-d\nMLP Head",        FUSE_COLOR,   "Fusion"),
            ("Softmax\nThreshold",             GREEN,        "Classify"),
            ("Grad-CAM\nOverlay",              ORANGE,       "Explain"),
            ("P(PNEUMONIA)\nUncertainty σ",    PURPLE,       "Output"),
        ]

        boxes  = VGroup()
        labels = VGroup()
        COLS   = 4

        for i, (lbl, color, tag) in enumerate(pipeline):
            row = i // COLS
            col = i %  COLS
            blk = make_block(lbl, width=2.5, height=1.1,
                              fill=PANEL, color=color)
            x = -4.6 + col * 3.1
            y = 1.5  - row * 2.0
            blk.move_to(RIGHT * x + UP * y)
            tag_txt = small_text(tag, color=color)
            tag_txt.next_to(blk, DOWN, buff=0.12)
            boxes.add(blk)
            labels.add(tag_txt)

        # Arrows
        pipe_arrows = VGroup()
        for i in range(len(boxes) - 1):
            if (i + 1) % COLS == 0 and i < len(boxes) - 1:
                # Row wrap: arrow goes down
                a = make_arrow(boxes[i][0].get_bottom(),
                               boxes[i][0].get_bottom() + DOWN * 0.4)
                pipe_arrows.add(a)
            elif i % COLS == COLS - 1:
                pass  # handled by wrap arrow above
            else:
                a = make_arrow(boxes[i][0].get_right(),
                               boxes[i + 1][0].get_left(), color=MUTED)
                pipe_arrows.add(a)

        for blk, lbl in zip(boxes, labels):
            self.play(FadeIn(blk, scale=0.92), FadeIn(lbl), run_time=0.28)

        for a in pipe_arrows:
            self.play(GrowArrow(a), run_time=0.2)

        self.wait(1.0)

        # ── Final result box ───────────────────────────────────────────────────
        result_bg = make_card(7.5, 1.5, fill=PANEL, stroke=GREEN)
        result_bg.move_to(DOWN * 3.4)

        result_txt = Text(
            "PNEUMONIA  |  P = 0.982  |  σ = 0.018  |  Grad-CAM: lower right lobe",
            font="Monospace", font_size=16, color=GREEN,
        )
        result_txt.move_to(result_bg)

        self.play(FadeIn(result_bg), Write(result_txt))
        self.wait(2.5)

        # ── Credits ───────────────────────────────────────────────────────────
        self.play(FadeOut(Group(*self.mobjects)))

        credits = VGroup(
            Text("MedFusionNet", font="Monospace",
                 font_size=44, color=BRAND_LIGHT, weight=BOLD),
            Text("Swin Transformer + DenseNet-121", font="Monospace",
                 font_size=22, color=MUTED),
            Text("AUC 0.98  ·  Accuracy 93.3%  ·  Recall 97%",
                 font="Monospace", font_size=18, color=GREEN),
            Text("github.com/carteeeltheboss/DPR_PFA4IADO",
                 font="Monospace", font_size=16, color=BRAND),
        )
        credits.arrange(DOWN, buff=0.35)

        for line in credits:
            self.play(FadeIn(line, shift=UP * 0.2), run_time=0.55)
        self.wait(3.0)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.5)
