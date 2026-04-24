"""
utils/styles.py
Shared palette, typography helpers, and reusable Manim mobjects
for the MedFusionNet animation series.
"""
from __future__ import annotations
from manim import *

# ── Colour palette ─────────────────────────────────────────────────────────────
BG          = "#0D1117"   # near-black background
PANEL       = "#161B22"   # card / panel fill
BORDER      = "#30363D"   # subtle border
WHITE       = "#F0F6FC"
MUTED       = "#8B949E"

BRAND       = "#2563EB"   # primary blue  (matches Flask UI)
BRAND_LIGHT = "#60A5FA"
GREEN       = "#22C55E"
YELLOW      = "#F59E0B"
RED         = "#EF4444"
PURPLE      = "#A855F7"
TEAL        = "#14B8A6"
ORANGE      = "#F97316"

# Branch colours
DENSE_COLOR = TEAL      # DenseNet-121 branch
SWIN_COLOR  = PURPLE    # Swin Transformer branch
FUSE_COLOR  = BRAND     # Fusion head


# ── Typography helpers ─────────────────────────────────────────────────────────
def title_text(t: str, color=WHITE, **kwargs) -> Text:
    return Text(t, font="Monospace", color=color, font_size=36, **kwargs)

def body_text(t: str, color=WHITE, **kwargs) -> Text:
    return Text(t, font="Monospace", color=color, font_size=22, **kwargs)

def small_text(t: str, color=MUTED, **kwargs) -> Text:
    return Text(t, font="Monospace", color=color, font_size=16, **kwargs)

def label_text(t: str, color=MUTED, **kwargs) -> Text:
    return Text(t, font="Monospace", color=color, font_size=14, **kwargs)

def code_text(t: str, color=BRAND_LIGHT, **kwargs) -> Text:
    return Text(t, font="Monospace", color=color, font_size=18, **kwargs)


# ── Reusable mobjects ──────────────────────────────────────────────────────────
def make_card(width: float, height: float, fill=PANEL,
              stroke=BORDER, stroke_width=1.5) -> RoundedRectangle:
    """Rounded rectangle card (use as background panel)."""
    r = RoundedRectangle(corner_radius=0.12, width=width, height=height)
    r.set_fill(fill, opacity=1).set_stroke(stroke, width=stroke_width)
    return r


def make_block(label: str, width=2.4, height=0.9, fill=PANEL,
               color=BORDER, text_color=WHITE) -> VGroup:
    """A labelled rectangular block used to represent layers."""
    rect = Rectangle(width=width, height=height)
    rect.set_fill(fill, opacity=1).set_stroke(color, width=2)
    txt  = Text(label, font="Monospace", font_size=16, color=text_color)
    txt.move_to(rect.get_center())
    return VGroup(rect, txt)


def make_arrow(start, end, color=MUTED, tip_length=0.18) -> Arrow:
    return Arrow(start, end, color=color, tip_length=tip_length,
                 stroke_width=2.5, buff=0.08)


def make_label_above(mob, text: str, color=MUTED, size=14) -> Text:
    lbl = Text(text, font="Monospace", font_size=size, color=color)
    lbl.next_to(mob, UP, buff=0.15)
    return lbl


def make_probability_bar(label: str, value: float, width=4.0,
                         bar_color=BRAND, bg_color=BORDER) -> VGroup:
    """Horizontal probability bar (value in [0,1])."""
    track = Rectangle(width=width, height=0.35)
    track.set_fill(bg_color, opacity=1).set_stroke(BORDER, width=1)
    fill  = Rectangle(width=max(0.01, width * value), height=0.35)
    fill.set_fill(bar_color, opacity=1).set_stroke(color=bar_color, width=0)
    fill.align_to(track, LEFT)
    lbl   = Text(label, font="Monospace", font_size=16, color=WHITE)
    lbl.next_to(track, LEFT, buff=0.2)
    pct   = Text(f"{value*100:.1f}%", font="Monospace", font_size=16, color=WHITE)
    pct.next_to(track, RIGHT, buff=0.2)
    return VGroup(track, fill, lbl, pct)


def make_section_title(text: str) -> VGroup:
    """Full-width section title with underline."""
    t = Text(text, font="Monospace", font_size=30, color=WHITE)
    line = Line(LEFT * 6, RIGHT * 6, color=BRAND, stroke_width=1.5)
    line.next_to(t, DOWN, buff=0.12)
    return VGroup(t, line)
