"""utils/styles.py — shared palette and helpers (no font= args for portability)."""
from __future__ import annotations
from manim import *
import numpy as np

BG          = "#0D1117"
PANEL       = "#161B22"
BORDER      = "#30363D"
WHITE       = "#F0F6FC"
MUTED       = "#8B949E"
BRAND       = "#2563EB"
BRAND_LIGHT = "#60A5FA"
GREEN       = "#22C55E"
YELLOW      = "#F59E0B"
RED         = "#EF4444"
PURPLE      = "#A855F7"
TEAL        = "#14B8A6"
ORANGE      = "#F97316"
DENSE_COLOR = TEAL
SWIN_COLOR  = PURPLE
FUSE_COLOR  = BRAND

def T(text, size=22, color=WHITE, **kw):
    try:
        return Text(text, font_size=size, color=color, **kw)
    except TypeError:
        return Text(text, color=color, **kw).scale(size / 48)

def make_card(w, h, fill=PANEL, stroke=BORDER, sw=1.5):
    r = RoundedRectangle(corner_radius=0.1, width=w, height=h)
    r.set_fill(fill, opacity=1).set_stroke(stroke, width=sw)
    return r

def pixel_color(value_0_255):
    """Convert 0-255 greyscale to a hex colour for visualisation."""
    v = int(np.clip(value_0_255, 0, 255))
    return "#{:02x}{:02x}{:02x}".format(v, v, v)

def norm_color(value, vmin=-1.6, vmax=1.6):
    """Map a normalized value to a blue-white-red colormap."""
    t = (value - vmin) / (vmax - vmin)
    t = float(np.clip(t, 0, 1))
    if t < 0.5:
        s = t * 2
        r = int(30  + s * (240 - 30))
        g = int(100 + s * (240 - 100))
        b = int(255 + s * (255 - 255))
    else:
        s = (t - 0.5) * 2
        r = int(240 + s * (255 - 240))
        g = int(240 - s * (240 - 30))
        b = int(255 - s * (255 - 30))
    return "#{:02x}{:02x}{:02x}".format(r, g, b)
