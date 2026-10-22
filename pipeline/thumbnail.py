"""Closed Tabs thumbnails: someone at 3 a.m. and the thing looking back at them.

The reference is the cover that sold 8.1 million views on "62 SCARIEST TikToks":
one shocked face, one pale wrong thing, a camcorder frame around both, and
almost no words. The recipe, every episode:

  - FLUX draws a frightened person, 20-35, close in on the left of the frame -
    eyes wide, mouth open, turned toward the right and not toward the lens. The
    person changes every time (seeded from the episode). On the right is one
    blank, evenly lit opening: a monitor, a window or a doorway, generated empty.
  - Gemini gives the rough corners of that blank opening; code then finds the lit
    blank panel itself in the pixels (refine_quad) and paints the figure into
    exactly that shape. A candidate whose opening cannot be found is ranked below
    every one whose opening was.
  - In the opening stands the figure: pale, hair over its face, two hollow eyes.
    It is drawn here rather than generated, so it never comes out with six
    fingers, never spells anything, is nobody real, and cannot wander into blood
    or a body - only a shape nobody wants in their window.
  - Two or three words at most, in Anton, white with one word red and a heavy
    black outline. An episode may also carry none; the picture is the hook.
  - A camcorder viewfinder over all of it: corner brackets, REC and the 03:14 AM
    stamp together in the top right, a light VHS layer.
  - The bottom right quarter stays empty. YouTube prints the duration badge there.

Six candidates are drawn. Gemini rejects hands without five fingers, broken
faces, eyes on the lens, lettering and anything graphic; the words must still
read at 336x189, the size a thumbnail is actually seen at in search results;
the best survivor wins.

The weekly round-up ("5 Most Disturbing...") wears the same frame around a
collage of four to six captioned stills instead: `collage`.
"""
import random
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from core import fonts
from core.logger import get_logger

log = get_logger("thumbnail")

W, H = 1280, 720
SMALL = (336, 189)
CANDIDATES = 6
# YouTube prints the duration badge over this corner. Nothing that carries the
# cover may sit in it - not a word, not the face, not the figure.
SAFE = (0.75, 0.80, 1.0, 1.0)
# And the corner the viewfinder's own REC and timecode take, in pixels, which
# nothing else may be laid into either.
OSD = (W - 340, 0, W, 210)

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

# Three openings, not one monitor: a window and a doorway carry the same dread
# and stop ten episodes in a row looking like the same photograph. Every one of
# them is asked for blank and evenly lit, because refine_quad finds the figure's
# place by looking for exactly that - a bright, smooth, nearly grey panel.
OPENINGS = [
    "a large flat computer monitor filling the right half of the frame, square to the "
    "camera, its screen a flat rectangle parallel to the image, the screen blank and "
    "evenly lit pale grey-white, completely empty",
    "a tall window filling the right half of the frame, square to the camera, its glass "
    "a flat rectangle parallel to the image, the glass blank and evenly lit pale "
    "grey-white, completely empty",
    "an open doorway filling the right half of the frame, square to the camera, the "
    "opening a flat rectangle parallel to the image, full of blank evenly lit pale "
    "grey-white light, completely empty",
]


def base_prompts(seed, n=CANDIDATES):
    rng = random.Random(seed)
    who = rng.sample(PEOPLE, min(n, len(PEOPLE)))
    return [
        f"{p}, in their twenties or thirties, terrified, eyes wide open, mouth open in a "
        f"gasp, head and shoulders large in the left third of the frame, close up, seen in "
        f"half profile, staring at {OPENINGS[(seed + i) % len(OPENINGS)]}, the person looks "
        f"at it and never at the camera, a dark room at night lit by teal and magenta neon "
        f"and the pale glow of the opening, well exposed, colourful, cinematic, "
        f"photorealistic, 35mm film still, "
        f"no text, no letters, no logos, five fingers on each hand"
        for i, p in enumerate(who)]


# --- what stands in the opening ---------------------------------------------------

def screen_figure(seed, size=(1200, 900)):
    """A pale face in the dark, hair closed around it, two hollow eyes, a mouth open.

    Drawn rather than generated, deliberately: it has no fingers to get wrong, it
    spells nothing, it is nobody real, and it cannot wander into blood or a body.

    It is drawn at 240 px and blown up, with grain, scanlines and a dropped line
    over the top. That is the whole trick: simple shapes at that size read as
    something a bad camera caught, where the same shapes drawn sharp read as a
    cartoon - the first four attempts at this figure all came out as a cartoon.
    The hair is one mass exactly the width of the head with the face cut out of
    it, never two shapes beside the head; shapes beside a head read as ears, and
    every version that had them looked like an animal.
    """
    w, h = size
    rng = np.random.default_rng(seed)
    lw, lh = 240, max(60, round(240 * h / w))
    t = np.linspace(0, 1, lh)[:, None]
    back = (1 - t) * np.array([66, 74, 72.0]) + t * np.array([24, 29, 28.0])
    im = Image.fromarray(np.repeat(back[:, None, :], lw, axis=1).astype(np.uint8))

    R = lh * 0.19
    cx = lw * (0.44 + 0.12 * rng.random())
    cy = lh * 0.44
    tilt = (rng.random() - 0.5) * R * 0.2            # never quite square to the lens

    # head and shoulders, pale, and the light they throw back into the room
    body = Image.new("L", (lw, lh), 0)
    bd = ImageDraw.Draw(body)
    bd.polygon([(cx - R * 2.0, lh), (cx - R * 0.78, cy + R * 1.15),
                (cx + R * 0.78, cy + R * 1.15), (cx + R * 2.0, lh)], fill=92)
    bd.ellipse((cx - R * 0.80, cy - R * 1.20, cx + R * 0.80, cy + R * 1.22), fill=188)
    body = body.filter(ImageFilter.GaussianBlur(lh * 0.016))
    pale = Image.new("RGB", (lw, lh), (202, 212, 208))
    im.paste(pale, (0, 0), body)
    glow = Image.new("RGB", (lw, lh), (0, 0, 0))
    glow.paste(pale, (0, 0), body.filter(ImageFilter.GaussianBlur(lh * 0.09)))
    im = Image.fromarray(np.clip(np.asarray(im, np.int16)
                                 + np.asarray(glow, np.int16) // 4, 0, 255).astype(np.uint8))

    # the hair: one mass, the face cut out of it
    outer = Image.new("L", (lw, lh), 0)
    ImageDraw.Draw(outer).ellipse((cx - R * 0.86 + tilt, cy - R * 1.36,
                                   cx + R * 0.86 + tilt, cy + R * 2.6), fill=255)
    window = Image.new("L", (lw, lh), 0)
    ImageDraw.Draw(window).ellipse((cx - R * 0.52 + tilt, cy - R * 0.82,
                                    cx + R * 0.52 + tilt, cy + R * 1.10), fill=255)
    im.paste(Image.new("RGB", (lw, lh), (19, 23, 25)), (0, 0),
             ImageChops.subtract(outer, window).filter(ImageFilter.GaussianBlur(lh * 0.012)))

    # a brow and a jaw in shadow, so the face is not a flat white oval
    shade = Image.new("L", (lw, lh), 0)
    sd = ImageDraw.Draw(shade)
    sd.ellipse((cx - R * 0.48 + tilt, cy - R * 0.74, cx + R * 0.48 + tilt, cy - R * 0.16), fill=105)
    sd.ellipse((cx - R * 0.40 + tilt, cy + R * 0.64, cx + R * 0.40 + tilt, cy + R * 1.18), fill=115)
    im.paste(Image.new("RGB", (lw, lh), (24, 30, 32)), (0, 0),
             shade.filter(ImageFilter.GaussianBlur(lh * 0.020)))

    # the eyes are sockets, not dots: a dark core inside a softer hollow
    face = Image.new("L", (lw, lh), 0)
    ed = ImageDraw.Draw(face)
    for s in (-1, 1):
        ex = cx + tilt + s * R * 0.25
        dy = R * 0.04 * rng.random() * s
        ed.ellipse((ex - R * 0.18, cy - R * 0.22 + dy, ex + R * 0.18, cy + R * 0.32 + dy), fill=120)
        ed.ellipse((ex - R * 0.10, cy - R * 0.08 + dy, ex + R * 0.10, cy + R * 0.22 + dy), fill=255)
    ed.ellipse((cx + tilt - R * 0.12, cy + R * 0.50, cx + tilt + R * 0.12, cy + R * 1.00), fill=140)
    ed.ellipse((cx + tilt - R * 0.07, cy + R * 0.58, cx + tilt + R * 0.07, cy + R * 0.94), fill=255)
    im.paste(Image.new("RGB", (lw, lh), (9, 12, 14)), (0, 0),
             face.filter(ImageFilter.GaussianBlur(lh * 0.011)))

    # blown up, then the tape it came off: grain, scanlines, one dropped line
    im = im.resize(size, Image.BICUBIC).filter(ImageFilter.GaussianBlur(h * 0.004))
    a = np.asarray(im, np.int16) + rng.integers(-18, 19, (h, w, 1))
    a[::3] = (a[::3] * 0.88).astype(np.int16)
    tear = int(rng.integers(0, max(1, h - 4)))
    a[tear:tear + 3] = np.roll(a[tear:tear + 3], int(rng.integers(-14, 15)), axis=1)
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx / w - 0.5) * 1.1, yy / h - 0.5) / 0.72
    a = a * np.clip(1.05 - r ** 2 * 0.62, 0.34, 1.0)[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))


def refine_quad(base, quad):
    """The blank lit panel itself, found in the pixels near Gemini's rough corners.

    The opening was generated as an evenly lit pale panel: bright, nearly grey and
    smooth. The largest such region overlapping the rough quad is the opening; its
    four extreme points are the corners. Returns a quad in 0-1 or None."""
    from scipy import ndimage
    a = np.asarray(base.convert("RGB"), np.float32)
    v = a.max(axis=2)
    sat = (v - a.min(axis=2)) / (v + 1e-3)
    grey = a.mean(axis=2)
    smooth = ndimage.uniform_filter((grey - ndimage.uniform_filter(grey, 9)) ** 2, 9) ** 0.5
    mask = (v > np.percentile(v, 80)) & (v > 120) & (sat < 0.35) & (smooth < 6)
    mask = ndimage.binary_opening(mask, iterations=2)
    mask = ndimage.binary_closing(mask, iterations=3)
    lab, n = ndimage.label(mask)
    if not n:
        return None
    h, w = mask.shape
    if quad and valid_quad(quad):
        q = np.array(quad, float) * [w, h]
        x0, y0 = q.min(axis=0)
        x1, y1 = q.max(axis=0)
        rough_area = (x1 - x0) * (y1 - y0)
    else:
        x0, y0, x1, y1 = w * 0.45, 0, w, h * 0.9          # the right half, where it is asked for
        rough_area = None
    best, best_n = None, 0
    for k in range(1, n + 1):
        ys, xs = np.nonzero(lab[int(y0):int(y1) + 1, int(x0):int(x1) + 1] == k)
        if len(xs) > best_n:
            best, best_n = k, len(xs)
    if not best:
        return None
    ys, xs = np.nonzero(lab == best)
    area = len(xs)
    if area < 0.04 * w * h:
        return None
    if rough_area and not 0.45 * rough_area <= area <= 1.6 * rough_area:
        return None
    # An opening is a convex quadrilateral that its pixels nearly fill.
    tl = (xs + ys).argmin(); br = (xs + ys).argmax()
    tr = (xs - ys).argmax(); bl = (xs - ys).argmin()
    pts = np.array([[xs[tl], ys[tl]], [xs[tr], ys[tr]], [xs[br], ys[br]], [xs[bl], ys[bl]]], float)
    quad_area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
                          - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
    if quad_area <= 0 or area / quad_area < 0.85:
        return None
    # pull in by 1% toward the centre so the paint stays inside the frame or bezel
    c = pts.mean(axis=0)
    pts = c + (pts - c) * 0.995
    out = [[round(float(x / w), 4), round(float(y / h), 4)] for x, y in pts]
    return out if valid_quad(out) else None


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
    # the opening glows: a soft bloom of its own light around the quad
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
    area = 0.5 * abs(np.dot(pts[:, 0], np.roll(pts[:, 1], 1))
                     - np.dot(pts[:, 1], np.roll(pts[:, 0], 1)))
    return (all(c > 0 for c in cross) or all(c < 0 for c in cross)) and area > 0.04


# --- words, the viewfinder, the checks --------------------------------------------

def _words(text):
    """Three words at most, or none at all. The reference cover sells the picture."""
    return [w for w in re.findall(r"[\w'?!.-]+", (text or "").upper())][:3]


def safe_clear(box):
    """Does `box` stay out of the corner YouTube prints the duration badge over?"""
    if not box:
        return True
    _, _, x1, y1 = box
    return not (x1 > SAFE[0] * W and y1 > SAFE[1] * H)


def draw_words(im, text, face=None, box=None):
    """Anton, white with the last word red, black outline, on the left half - above
    or below the face, whichever leaves it clear, and never into the badge corner.
    `box` is (x, y, width, height) when the caller has somewhere exact in mind, as
    the round-up cover does with its title band.
    Returns (box, size), or (None, 0) when the episode carries no words."""
    words = _words(text)
    if not words:
        return None, 0
    d = ImageDraw.Draw(im)
    x0, y0, box_w, box_h = box or (48, 44, W * 0.52, H * 0.38)
    size, lines = fonts.fit(" ".join(words), "title", box_w, box_h, 190, min_size=60, spacing=1.0)
    f = fonts.font("title", size)
    block_h = size * len(lines)
    if box is None and face and isinstance(face, list) and len(face) == 4:
        fy0, fy1 = face[1] * H, face[3] * H
        if fy0 < 44 + block_h + 20:                 # the top would cover the face
            # the bottom, but no lower than the badge corner allows
            low = min(int(H - block_h - 50), int(SAFE[1] * H - block_h - 10))
            y0 = low if fy1 < low - 20 else 44
    stroke = max(6, size // 14)
    red_word = words[-1]
    y, widest = y0, 0
    for line in lines:
        x = x0
        for word in line.split():
            col = (232, 44, 52) if word == red_word else (255, 255, 255)
            d.text((x, y), word, font=f, fill=col, stroke_width=stroke, stroke_fill=(0, 0, 0))
            x += f.getlength(word + " ")
        widest = max(widest, x - x0)
        y += size * 1.0
    return (x0, y0, x0 + widest, y), size


def viewfinder(im, seed=0):
    """The camcorder laid over the picture: VHS slip, grain and scanlines, four
    corner brackets, and REC with the 03:14 AM stamp together in the top right.
    Nothing but a bracket goes in the bottom right - the duration badge lands there."""
    rng = np.random.default_rng(seed)
    a = np.asarray(im, np.int16).copy()
    a[:, 3:, 0] = a[:, :-3, 0]                         # red channel slip
    a[:, :-3, 2] = a[:, 3:, 2]                         # blue channel slip
    a += rng.integers(-9, 10, a.shape[:2], dtype=np.int16)[..., None]
    a[::3] = (a[::3] * 0.88).astype(np.int16)          # scanlines
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(out)

    inset_x, inset_y, arm, width = int(W * 0.028), int(H * 0.042), int(W * 0.045), 5
    for x, y, sx, sy in ((inset_x, inset_y, 1, 1), (W - inset_x, inset_y, -1, 1),
                         (inset_x, H - inset_y, 1, -1), (W - inset_x, H - inset_y, -1, -1)):
        d.line((x, y, x + sx * arm, y), fill=(238, 242, 246), width=width)
        d.line((x, y, x, y + sy * arm), fill=(238, 242, 246), width=width)

    f = fonts.font("label", 54)
    right = W - 48
    d.text((right - f.getlength("REC"), 78), "REC", font=f, fill=(255, 255, 255),
           stroke_width=2, stroke_fill=(0, 0, 0))
    dot = right - f.getlength("REC") - 44
    d.ellipse((dot, 88, dot + 28, 116), fill=(232, 30, 30))
    stamp = "03:14 AM"
    d.text((right - f.getlength(stamp), 140), stamp, font=f, fill=(255, 255, 255),
           stroke_width=2, stroke_fill=(0, 0, 0))
    return out


def readable_small(im, text_size, box):
    """Do the words survive at 336x189? Cap height >= 16 px and real contrast where
    they were actually drawn. A cover with no words has nothing to fail."""
    if not box or not text_size:
        return True, {"cap_px": 0.0, "spread": 0.0}
    scale = SMALL[1] / H
    cap_px = text_size * 0.72 * scale
    small = np.asarray(im.resize(SMALL, Image.LANCZOS).convert("L"), float)
    x0, y0, x1, y1 = box
    region = small[int(y0 * scale):max(int(y1 * scale), int(y0 * scale) + 2),
                   int(x0 * scale):max(int(x1 * scale), int(x0 * scale) + 2)]
    spread = np.percentile(region, 95) - np.percentile(region, 5)
    return bool(cap_px >= 16 and spread >= 120), {"cap_px": round(float(cap_px), 1),
                                                  "spread": round(float(spread), 1)}


def compose(base_path, quad, words, seed, face=None):
    """Returns (image, text size, word box, whether the opening was found)."""
    from pipeline import grade
    base = Image.open(base_path).convert("RGB")
    s_ = max(W / base.width, H / base.height)
    base = base.resize((round(base.width * s_), round(base.height * s_)), Image.LANCZOS)
    x, y = (base.width - W) // 2, (base.height - H) // 2
    base = base.crop((x, y, x + W, y + H))
    # The rough quad from Gemini is in the original frame; the crop above is a
    # centred fill, so map it before refining.
    if quad and valid_quad(quad):
        bw, bh = base.width + 2 * x, base.height + 2 * y
        quad = [[(qx * bw - x) / W, (qy * bh - y) / H] for qx, qy in quad]
    exact = refine_quad(base, quad)
    base = grade.grade(base)
    if exact:
        q = np.array(exact) * [W, H]
        qw = np.linalg.norm(q[1] - q[0])
        qh = np.linalg.norm(q[3] - q[0])
        cw = 1200
        content = screen_figure(seed, (cw, max(400, int(cw * qh / max(qw, 1)))))
        base = paste_screen(base, exact, content)
    else:
        panel = screen_figure(seed, (720, 540)).resize((560, 420), Image.LANCZOS)
        base.paste(panel, (W - 600, 190))
    box, size = draw_words(base, words, face)
    return viewfinder(base, seed), size, box, bool(exact)


def choose(bases, verdicts, words, out_dir, seed):
    """Compose every approved base, keep those that read small, return the best.

    Returns (path of the winner, report). Falls back to the best-scored base when
    none passes everything, and says so - a thumbnail is never left out."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report, ranked = [], []
    for k, b in enumerate(bases):
        v = verdicts.get(Path(b), {})
        im, size, box, exact = compose(b, v.get("screen"), words, seed + k, v.get("face"))
        path = out_dir / f"thumb_{k}.jpg"
        im.save(path, quality=92)
        readable, m = readable_small(im, size, box)
        clear = safe_clear(box)
        row = {"file": path.name, "ok": v.get("ok", False), "defect": v.get("defect"),
               "readable": readable, "badge_corner_clear": clear, "opening_found": exact,
               **m, "score": v.get("score", 0)}
        report.append(row)
        if row["ok"] and readable and clear:
            ranked.append((exact, row["score"], path))
    if ranked:
        best = max(ranked, key=lambda r: (r[0], r[1]))[2]
    else:
        best = out_dir / max(report, key=lambda r: (r["opening_found"], r["readable"],
                                                    r["badge_corner_clear"], r["score"]))["file"]
        log.warning("no candidate passed every check; using %s", best.name)
    final = out_dir / "thumbnail.jpg"
    Image.open(best).save(final, quality=92, optimize=True)
    if final.stat().st_size > 2_000_000:
        Image.open(best).save(final, quality=80, optimize=True)
    return final, report


# --- the weekly round-up ----------------------------------------------------------

# Four to six frames. Five reads better as 3 over 2 than as a row with a hole in it.
ROWS = {4: [2, 2], 5: [3, 2], 6: [3, 3]}


def collage(frames, captions, words, out_path, seed=0):
    """The round-up cover ("5 Most Disturbing..."): four to six stills with captions,
    under the same viewfinder as every other Closed Tabs thumbnail.

    The title takes a band across the top and each caption sits at the top of its
    own cell, so the bottom right cell keeps its quarter clear for the badge."""
    from pipeline import grade
    frames, captions = list(frames), list(captions)
    if not 4 <= len(frames) <= 6:
        raise ValueError(f"a round-up cover takes four to six frames, got {len(frames)}")
    captions += [""] * (len(frames) - len(captions))
    sheet = Image.new("RGB", (W, H), (10, 12, 16))
    # The title gets a band of its own across the top rather than the half-frame
    # block a single episode uses; without one it sat over the first row of cells.
    band = int(H * 0.20) if _words(words) else 0
    gap = 6
    rows = ROWS[len(frames)]
    ch = (H - band - gap * (len(rows) + 1)) // len(rows)
    k, y = 0, band + gap
    for cols in rows:
        cw = (W - gap * (cols + 1)) // cols
        x = gap
        for _ in range(cols):
            cell = grade.grade(Image.open(frames[k]).convert("RGB"))
            s_ = max(cw / cell.width, ch / cell.height)
            cell = cell.resize((round(cell.width * s_), round(cell.height * s_)), Image.LANCZOS)
            cx, cy = (cell.width - cw) // 2, (cell.height - ch) // 2
            cell = cell.crop((cx, cy, cx + cw, cy + ch))
            cd = ImageDraw.Draw(cell)
            cd.rectangle((0, 0, cw - 1, ch - 1), outline=(70, 78, 92), width=2)
            cap = captions[k].strip().upper()
            if cap:
                # Captions ride at the top of their cell, which keeps the bottom
                # right one out of the badge corner. The one cell the REC cluster
                # reaches into takes its caption along the bottom instead - the top
                # row's right-hand caption ran straight through the 03:14 AM stamp.
                bar = int(ch * 0.19)
                top = not (y < OSD[3] and x + cw > OSD[0])
                y_bar = 0 if top else ch - bar
                cd.rectangle((0, y_bar, cw, y_bar + bar), fill=(12, 14, 20))
                cs, cl = fonts.fit(cap, "label", cw - 24, ch * 0.15, int(ch * 0.15), min_size=18)
                cy_ = y_bar + int(ch * 0.02)
                for line in cl:
                    cd.text((12, cy_), line, font=fonts.font("label", cs), fill=(255, 255, 255),
                            stroke_width=2, stroke_fill=(0, 0, 0))
                    cy_ += cs
            sheet.paste(cell, (x, y))
            x += cw + gap
            k += 1
        y += ch + gap
    if band:
        # clear of the REC cluster, which the viewfinder puts in the top right
        draw_words(sheet, words, box=(48, 14, OSD[0] - 90, band - 24))
    out = viewfinder(sheet, seed)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path, quality=92, optimize=True)
    if out_path.stat().st_size > 2_000_000:
        out.save(out_path, quality=80, optimize=True)
    return out_path
