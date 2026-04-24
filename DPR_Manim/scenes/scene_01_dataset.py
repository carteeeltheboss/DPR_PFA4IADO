"""
scenes/scene_01_dataset.py
Scene 1 — Dataset Preparation
Covers: folder structure, class counts, label encoding, one X-ray entering the pipeline.
Duration: ~60 s
"""
from __future__ import annotations
from manim import *
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from utils.styles import *


class Scene01Dataset(Scene):
    """Dataset structure, class counts, and label encoding."""

    def construct(self):
        self.camera.background_color = BG

        # ── 1. Title ───────────────────────────────────────────────────────────
        title = make_section_title("Phase 1 — Dataset Preparation")
        self.play(FadeIn(title, shift=DOWN * 0.3))
        self.wait(1.2)
        self.play(FadeOut(title))

        # ── 2. Dataset overview card ───────────────────────────────────────────
        dataset_card = make_card(11, 1.2)
        dataset_card.move_to(UP * 3)
        ds_label = Text(
            "paultimothymooney / chest-xray-pneumonia  (Kaggle)",
            font="Monospace", font_size=18, color=BRAND_LIGHT,
        )
        ds_label.move_to(dataset_card)
        self.play(FadeIn(dataset_card), Write(ds_label))
        self.wait(0.8)

        # ── 3. Folder tree ────────────────────────────────────────────────────
        splits = ["train", "val", "test"]
        classes = ["NORMAL", "PNEUMONIA"]
        counts = {
            "train": {"NORMAL": 1341, "PNEUMONIA": 3875},
            "val":   {"NORMAL": 8,    "PNEUMONIA": 8},
            "test":  {"NORMAL": 234,  "PNEUMONIA": 390},
        }
        split_colors = {
            "train":   GREEN,
            "val":     YELLOW,
            "test":    ORANGE,
        }
        class_colors  = {"NORMAL": TEAL, "PNEUMONIA": RED}

        tree_items = VGroup()
        y = 1.5
        for split in splits:
            # split row
            split_txt = Text(
                f"  {split}/",
                font="Monospace", font_size=20,
                color=split_colors[split],
            )
            split_txt.move_to(LEFT * 3.5 + UP * y)
            tree_items.add(split_txt)
            y -= 0.52
            for cls in classes:
                n = counts[split][cls]
                cls_txt = Text(
                    f"      {cls}/    ({n:,} images)",
                    font="Monospace", font_size=16,
                    color=class_colors[cls],
                )
                cls_txt.move_to(LEFT * 2.8 + UP * y)
                tree_items.add(cls_txt)
                y -= 0.46

        folder_icon = Text("📁 chest_xray/", font="Monospace",
                           font_size=22, color=WHITE)
        folder_icon.move_to(LEFT * 4 + UP * 2.4)

        self.play(FadeIn(folder_icon))
        for item in tree_items:
            self.play(FadeIn(item, shift=RIGHT * 0.2), run_time=0.18)
        self.wait(1.2)

        # ── 4. Total counts banner ────────────────────────────────────────────
        total_normal    = sum(counts[s]["NORMAL"]    for s in splits)
        total_pneumonia = sum(counts[s]["PNEUMONIA"] for s in splits)
        total           = total_normal + total_pneumonia

        count_card = make_card(5, 2.2)
        count_card.move_to(RIGHT * 3.8 + UP * 0.8)

        c_title = Text("Total images", font="Monospace",
                       font_size=18, color=MUTED)
        c_title.move_to(count_card.get_top() + DOWN * 0.32)

        c_total = Text(f"{total:,}", font="Monospace",
                       font_size=38, color=WHITE)
        c_total.move_to(count_card)

        c_norm = Text(f"NORMAL      {total_normal:,}  ({100*total_normal/total:.0f}%)",
                      font="Monospace", font_size=16, color=TEAL)
        c_pneu = Text(f"PNEUMONIA   {total_pneumonia:,}  ({100*total_pneumonia/total:.0f}%)",
                      font="Monospace", font_size=16, color=RED)
        c_norm.next_to(c_total, DOWN, buff=0.18)
        c_pneu.next_to(c_norm,  DOWN, buff=0.14)

        self.play(FadeIn(count_card), Write(c_title))
        self.play(FadeIn(c_total))
        self.play(FadeIn(c_norm), FadeIn(c_pneu))
        self.wait(1.0)

        # ── 5. Class imbalance bar ────────────────────────────────────────────
        imbal_label = Text("Class imbalance", font="Monospace",
                           font_size=17, color=MUTED)
        imbal_label.next_to(count_card, DOWN, buff=0.25)

        track = Rectangle(width=4.8, height=0.45)
        track.set_fill(BORDER, opacity=1).set_stroke(BORDER, width=0)

        pneu_frac = total_pneumonia / total
        fill_pneu = Rectangle(width=4.8 * pneu_frac, height=0.45)
        fill_pneu.set_fill(RED, opacity=0.85).set_stroke(color=RED, width=0)
        fill_norm = Rectangle(width=4.8 * (1 - pneu_frac), height=0.45)
        fill_norm.set_fill(TEAL, opacity=0.85).set_stroke(color=TEAL, width=0)
        fill_pneu.align_to(track, RIGHT)
        fill_norm.align_to(track, LEFT)

        bar_group = VGroup(track, fill_norm, fill_pneu)
        bar_group.next_to(imbal_label, DOWN, buff=0.12)
        bar_group.move_to(bar_group.get_center() * UP + RIGHT * 3.8)

        self.play(FadeIn(imbal_label), FadeIn(track),
                  GrowFromEdge(fill_norm, LEFT),
                  GrowFromEdge(fill_pneu, RIGHT))
        self.wait(0.8)

        # ── 6. Label encoding ─────────────────────────────────────────────────
        self.play(
            FadeOut(folder_icon), FadeOut(tree_items),
            FadeOut(count_card), FadeOut(c_title), FadeOut(c_total),
            FadeOut(c_norm), FadeOut(c_pneu),
            FadeOut(imbal_label), FadeOut(bar_group),
            FadeOut(dataset_card), FadeOut(ds_label),
        )

        enc_title = make_section_title("Label Encoding")
        enc_title.move_to(UP * 2.8)
        self.play(FadeIn(enc_title))

        boxes = VGroup()
        for i, (name, lbl, color) in enumerate([
            ("NORMAL",    "→  0", TEAL),
            ("PNEUMONIA", "→  1", RED),
        ]):
            box = make_card(4.8, 1.5)
            box.move_to(LEFT * 2.8 + RIGHT * i * 5.8 + UP * 0.3)

            cls_txt = Text(name, font="Monospace",
                           font_size=26, color=color, weight=BOLD)
            lbl_txt = Text(lbl, font="Monospace",
                           font_size=28, color=WHITE)
            cls_txt.move_to(box.get_center() + UP * 0.25)
            lbl_txt.next_to(cls_txt, DOWN, buff=0.22)
            boxes.add(VGroup(box, cls_txt, lbl_txt))

        for b in boxes:
            self.play(FadeIn(b, scale=0.9), run_time=0.5)

        note = small_text(
            "Binary classification: model outputs logits for [NORMAL, PNEUMONIA].",
        )
        note.move_to(DOWN * 2.2)
        self.play(FadeIn(note))
        self.wait(1.5)

        # ── 7. One image entering the pipeline ────────────────────────────────
        self.play(FadeOut(boxes), FadeOut(note), FadeOut(enc_title))

        pipeline_title = make_section_title("One image enters the pipeline →")
        pipeline_title.move_to(UP * 2.8)
        self.play(FadeIn(pipeline_title))

        # Placeholder chest X-ray (grey rectangle simulating film)
        # ⬇ REPLACE THIS with ImageMobject("assets/sample_xray.jpeg") for real image
        xray_rect = Rectangle(width=2.4, height=2.4)
        xray_rect.set_fill("#1a1a2e", opacity=1)
        xray_rect.set_stroke(MUTED, width=1.5)
        # Simulate lung opacities with ellipses
        left_lung  = Ellipse(width=0.9, height=1.5, color="#2a2a4e").set_fill("#2a2a4e", opacity=1)
        right_lung = Ellipse(width=0.9, height=1.5, color="#2a2a4e").set_fill("#2a2a4e", opacity=1)
        opacity    = Ellipse(width=0.55, height=0.4, color=YELLOW).set_fill(YELLOW, opacity=0.4)
        left_lung.move_to(xray_rect.get_center() + LEFT * 0.55)
        right_lung.move_to(xray_rect.get_center() + RIGHT * 0.55)
        opacity.move_to(xray_rect.get_center() + RIGHT * 0.45 + DOWN * 0.15)

        xray = VGroup(xray_rect, left_lung, right_lung, opacity)
        xray.move_to(LEFT * 4.5 + DOWN * 0.3)

        label_pneumonia = Text("PNEUMONIA  →  1",
                               font="Monospace", font_size=18, color=RED)
        label_pneumonia.next_to(xray, DOWN, buff=0.22)

        arrow_in = make_arrow(xray.get_right(), xray.get_right() + RIGHT * 2.5)

        pipe_box = make_block("Preprocessing\n+ Model",
                              width=3.2, height=1.4,
                              fill=PANEL, color=BRAND)
        pipe_box.next_to(arrow_in.get_end(), RIGHT, buff=0.15)

        arrow_out = make_arrow(
            pipe_box[0].get_right(),
            pipe_box[0].get_right() + RIGHT * 2.5
        )

        pred_txt = Text("P(Pneumonia) = 0.98\n→ PNEUMONIA",
                        font="Monospace", font_size=18, color=RED)
        pred_txt.next_to(arrow_out.get_end(), RIGHT, buff=0.15)

        self.play(FadeIn(xray, shift=UP * 0.3), FadeIn(label_pneumonia))
        self.play(GrowArrow(arrow_in))
        self.play(FadeIn(pipe_box))
        self.play(GrowArrow(arrow_out))
        self.play(FadeIn(pred_txt))
        self.wait(2.0)

        self.play(FadeOut(Group(*self.mobjects)))
        self.wait(0.3)
