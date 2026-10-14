"""Closed Tabs thumbnails: someone at 3 a.m. pointing at the impossible thing.

The recipe, every episode:
  - FLUX draws a shocked person, 20-35, in half profile on the left, pointing at
    an old monitor and looking at it - never at the camera. The person changes every
    time (seeded from the episode). The monitor is generated blank.
  - Gemini finds the four corners of that blank screen, and code paints the
    episode's screen into it in perspective, with a red circle on the detail
    that should not be there.
  - Two or three words in Anton, white with one word red, heavy black outline -
    words that add to the title rather than repeat it.
  - A light VHS layer, a REC dot and a 03:14 AM stamp.

Six to eight candidates are drawn. Gemini rejects hands without five fingers,
broken faces and eyes on the lens; the words must still read at 336x189, the size
a thumbnail is actually seen at in search results; the best survivor wins.
"""
import json
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from core import fonts
from core.logger import get_logger

log = get_logger("thumbnail")

W, H = 1280, 720
SMALL = (336, 189)
CANDIDATES = 8

PEOPLE = [
    "a young woman with short dark hair in a grey hoodie",
    "a young man with curly hair and round glasses in a black t-shirt",
    "a woman in her late twenties with long red hair in a denim jacket",
    "a man around thirty with a short beard in a flannel shirt",
    "a young woman with braided hair in an oversized sweater",
    "a young man with a buzz cut in a dark green bomber jacket",
    "a woman in her thirties with a messy bun and headphones around her neck",
    "a young man with long straight hair tied back, in a white hoodie",
    "a woman in her early twenties with a fringe and a striped shirt",
    "a man in his late twenties with wavy hair and a leather jacket",
]


def base_prompts(seed, n=CANDIDATES):
    rng = random.Random(seed)
    who = rng.sample(PEOPLE, n)
    return [
        f"{p}, shocked, eyes wide and mouth open, seen in three-quarter profile on the left "
        f"third of the frame, pointing a finger at an old computer monitor on the right, "
        f"looking at the monitor and not at the camera, the monitor screen is blank, evenly "
        f"lit pale grey, completely empty, a dark bedroom at 3 a.m. lit only by the monitor, "
        f"cinematic, photorealistic, 35mm film still, shallow depth of field, "
        f"no text, no letters, no logos, five fingers on each hand"
        for p in who]


# --- the screen content ----------------------------------------------------------

def screen_content(heading, line, size=(1200, 900)):
    """What is on the monitor: a dark page with the impossible detail circled."""
    w, h = size
    im = Image.new("RGB", size, (14, 17, 24))
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, w, 70), fill=(28, 34, 46))
    for i, c in enumerate(((235, 90, 80), (240, 190, 70), (90, 200, 110))):
        d.ellipse((24 + i * 34, 24, 46 + i * 34, 46), fill=c)
    hs, hl = fonts.fit(heading, "sans_bold", w - 120, 170, 110)
    for k, t in enumerate(hl):
        d.text((60, 120 + k * hs * 1.15), t, font=fonts.font("sans_bold", hs), fill=(225, 232, 238))
    y = 120 + len(hl) * hs * 1.15 + 40
    # the "image" the post carries: a near-black tile with the glowing detail
    tile = (60, int(y), w - 60, h - 60)
    d.rectangle(tile, fill=(5, 6, 8), outline=(40, 46, 58), width=3)
    cx, cy = (tile[0] + tile[2]) / 2, (tile[1] + tile[3]) / 2
    ls, ll = fonts.fit(line, "mono", tile[2] - tile[0] - 80, (tile[3] - tile[1]) * 0.5, 72)
    for k, t in enumerate(ll):
        tw = fonts.font("mono", ls).getlength(t)
        d.text((cx - tw / 2, cy - len(ll) * ls * 0.6 + k * ls * 1.2), t,
               font=fonts.font("mono", ls), fill=(120, 240, 210))
    r = max(tile[3] - tile[1], 200) * 0.42
    d.ellipse((cx - r * 1.3, cy - r, cx + r * 1.3, cy + r), outline=(232, 44, 52), width=16)
    return im


def _perspective_coeffs(src, dst):
    """Coefficients for Image.transform(PERSPECTIVE) mapping dst quad onto src quad."""
    a = []
    for (x, y), (u, v) in zip(dst, src):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y])
    b = np.array(src, np.float64).reshape(8)
    return np.linalg.solve(np.array(a, np.float64), b).tolist()


def paste_screen(base, quad, content):
    """Warp `content` into the quad (four corners, clockwise from top left, 0-1)."""
    dst = [(x * W, y * H) for x, y in quad]
    cw, ch = content.size
    src = [(0, 0), (cw, 0), (cw, ch), (0, ch)]
    coeffs = _perspective_coeffs(src, dst)
    warped = content.transform((W, H), Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).polygon(dst, fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(1.2))
    out = base.copy()
    out.paste(warped, (0, 0), mask)
    # the screen glows: a soft bloom of its own light around the quad
    glow = Image.new("RGB", (W, H), (0, 0, 0))
    glow.paste(warped, (0, 0), mask)
    glow = glow.filter(ImageFilter.GaussianBlur(26))
    return Image.blend(out, Image.fromarray(np.clip(np.asarray(out, np.int16) +
                                                    np.asarray(glow, np.int16) // 3, 0, 255)
                                            .astype(np.uint8)), 0.8)


def valid_quad(q):
    if not (isinstance(q, list) and len(q) == 4):
        return False
    try:
        pts = np.array(q, float)
    except (TypeError, ValueError):
        return False
    if pts.shape != (4, 2) or (pts < 0).any() or (pts > 1).any():
        return False
    # convex and big enough to read
    cross = []
    for i in range(4):
        a, b, c = pts[i], pts[(i + 1) % 4], pts[(i + 2) % 4]
        cross.append(np.cross(b - a, c - b))
    area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1)) - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
    return (all(c > 0 for c in cross) or all(c < 0 for c in cross)) and area > 0.04


# --- words, VHS, checks ---------------------------------------------------------

def _words(text):
    words = [w for w in re.findall(r"[\w'?!.-]+", text.upper())][:3]
    return words or ["CLOSED", "TABS"]


def draw_words(im, text, face=None):
    """Anton, white with the last word red, black outline, on the left half - above
    or below the face, whichever leaves it clear. Returns its box and size."""
    words = _words(text)
    d = ImageDraw.Draw(im)
    box_w, box_h = W * 0.56, H * 0.40
    phrase = " ".join(words)
    size, lines = fonts.fit(phrase, "title", box_w, box_h, 190, min_size=60, spacing=1.0)
    f = fonts.font("title", size)
    block_h = size * len(lines)
    x0, y0 = 48, 40
    if face and isinstance(face, list) and len(face) == 4:
        fy0, fy1 = face[1] * H, face[3] * H
        if fy0 < 40 + block_h + 20:                 # the top would cover the face
            y0 = int(H - block_h - 50) if fy1 < H - block_h - 70 else 40
    stroke = max(6, size // 14)
    red_word = words[-1]
    y = y0
    for line in lines:
        x = x0
        for word in line.split():
            col = (232, 44, 52) if word == red_word else (255, 255, 255)
            d.text((x, y), word, font=f, fill=col, stroke_width=stroke, stroke_fill=(0, 0, 0))
            x += f.getlength(word + " ")
        y += size * 1.0
    return (x0, y0, x0 + box_w, y), size


def vhs(im, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(im, np.int16).copy()
    a[:, 3:, 0] = a[:, :-3, 0]                         # red channel slip
    a[:, :-3, 2] = a[:, 3:, 2]                         # blue channel slip
    a += rng.integers(-9, 10, a.shape[:2], dtype=np.int16)[..., None]
    a[::3] = (a[::3] * 0.88).astype(np.int16)          # scanlines
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)
    f = fonts.font("label", 54)
    d.ellipse((W - 250, 44, W - 222, 72), fill=(232, 30, 30))
    d.text((W - 208, 34), "REC", font=f, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    stamp = "03:14 AM"
    d.text((W - 48 - f.getlength(stamp), H - 92), stamp, font=f, fill=(255, 255, 255),
           stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def readable_small(im, text_size):
    """Do the words survive at 336x189? Cap height >= 16 px and real contrast."""
    scale = SMALL[1] / H
    cap_px = text_size * 0.72 * scale
    small = im.resize(SMALL, Image.LANCZOS).convert("L")
    region = np.asarray(small)[:int(SMALL[1] * 0.45), :int(SMALL[0] * 0.58)].astype(float)
    spread = np.percentile(region, 95) - np.percentile(region, 5)
    return cap_px >= 16 and spread >= 120, {"cap_px": round(cap_px, 1), "spread": round(float(spread), 1)}


def compose(base_path, quad, words, heading, line, seed, face=None):
    base = Image.open(base_path).convert("RGB")
    s = max(W / base.width, H / base.height)
    base = base.resize((round(base.width * s), round(base.height * s)), Image.LANCZOS)
    x, y = (base.width - W) // 2, (base.height - H) // 2
    base = base.crop((x, y, x + W, y + H))
    if quad and valid_quad(quad):
        base = paste_screen(base, quad, screen_content(heading, line))
    else:
        # no usable screen found: a flat inset on the right rather than nothing
        panel = screen_content(heading, line, (720, 540)).resize((560, 420), Image.LANCZOS)
        base.paste(panel, (W - 600, 220))
    _, size = draw_words(base, words, face)
    return vhs(base, seed), size


def choose(bases, verdicts, words, heading, line, out_dir, seed):
    """Compose every approved base, keep those that read small, return the best.

    Returns (path of the winner, report). Falls back to the best-scored base when
    none passes everything, and says so - a thumbnail is never left out."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report, ranked = [], []
    for k, b in enumerate(bases):
        v = verdicts.get(Path(b), {})
        im, size = compose(b, v.get("screen"), words, heading, line, seed + k, v.get("face"))
        path = out_dir / f"thumb_{k}.jpg"
        im.save(path, quality=92)
        readable, m = readable_small(im, size)
        row = {"file": path.name, "ok": v.get("ok", False), "defect": v.get("defect"),
               "readable": readable, **m, "score": v.get("score", 0)}
        report.append(row)
        if row["ok"] and readable:
            ranked.append((row["score"], path))
    if ranked:
        best = max(ranked)[1]
    else:
        best = out_dir / max(report, key=lambda r: (r["readable"], r["score"]))["file"]
        log.warning("no candidate passed every check; using %s", best.name)
    final = out_dir / "thumbnail.jpg"
    Image.open(best).save(final, quality=92, optimize=True)
    if final.stat().st_size > 2_000_000:
        Image.open(best).save(final, quality=80, optimize=True)
    return final, report
