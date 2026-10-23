"""Closed Tabs thumbnails: something coming out of the monitor, and the person it
is coming for.

The recipe, every episode:

  - FLUX draws the whole scene. On the right, a monitor with something breaking
    out of it: a pale elongated arm with long thin fingers reaching through the
    glass, or a distorted face pushing through the screen as through a membrane.
    The glass cracks, splinters and glitch shards fly into the room, the screen
    burns turquoise and red. On the left, close in, a person of 20-35 recoils -
    eyes wide, mouth open - looking at the monitor and never at the lens.
  - Nothing graphic: no blood, no wound, no weapon. Nobody real, and nobody's
    character either. The long black hair and white dress of the film everyone
    would name is written out of the prompt and rejected by the vision check.
  - Earlier versions painted the thing in the screen here in code. It read as
    cheap, because it was: flat shapes cannot stand next to a photograph. FLUX
    draws it now, and the code only frames it.
  - When one generation will not hold both halves, the person and the monitor
    are drawn as separate shots and joined here (`compose_pair`): the monitor
    takes the right of the frame behind a feathered edge, and its light spills
    back across the room so the two were lit by the same screen.
  - Over the photograph: three words at most in Anton, white with one word red
    and a heavy black outline, or none at all - the picture is the hook. Then a
    camcorder viewfinder - corner brackets, REC and the 03:14 AM stamp together
    in the top right, a light VHS layer.
  - The bottom right quarter stays empty. YouTube prints the duration badge there.

Eight candidates are drawn. Gemini rejects the human's hands and face when they
come out wrong, eyes on the lens, lettering, anything graphic, and anything that
copies a known film character - while leaving the creature alone, because the
creature is supposed to be wrong. The words must still read at 336x189, the size
a thumbnail is actually seen at in search results; the best survivor wins.

The weekly round-up ("5 Most Disturbing...") wears the same frame around a
collage of four to six captioned stills instead: `collage`.
"""
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
# YouTube prints the duration badge over this corner. Nothing that carries the
# cover may sit in it - not a word, not the face, not the creature's hand.
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

# Two things come out of the screen, alternating, so ten covers in a row are not
# ten photographs of the same arm. Both are asked for HUGE and bone-white against
# the dark bezel: at 336x189, the size a cover is chosen at, a thin grey arm the
# width of a finger is a smudge, and the whole cover rests on that one shape.
CREATURES = [
    "a huge long pale white arm with very long thin fingers bursting out of the "
    "monitor's cracked screen and reaching toward the camera, the arm and the wide-spread "
    "hand filling the right half of the frame, the hand large and close, far too long for "
    "any body, bone-white skin bright against the dark bezel",
    "a huge gaunt pale white face pushing out through the monitor's screen as if the glass "
    "were a thin membrane stretched over it, the face filling the whole screen and bulging "
    "into the room, mouth open, features smeared and wrong, no hair, bone-white skin "
    "bright against the dark bezel",
]

# FLUX.1-schnell runs without guidance, so there is no negative prompt to put
# these in; they are written into the prompt as things the scene is made of.
# "A ghost girl with long black hair in a white dress" is the one shape this
# channel must never draw: it is a named character from a film, and it is also
# what every generator reaches for when asked for something coming out of a
# screen. It is written out here and rejected again in the vision check.
CLEAN = ("no blood, no wound, no injury, no weapon, bloodless, "
         "no text, no letters, no logos, no watermark")
# only where a creature is actually in the shot
CREATURE_CLEAN = ("the creature is bald and grey with no hair, not a girl, not a child, "
                  "no long black hair, no white dress, no nightgown")
# and only where a person is
PERSON_CLEAN = "the person's own hands are ordinary with five fingers each"

# Bright, not murky. The first v3 prompt asked for "a dark bedroom lit only by that
# screen", and a cover that reads as a dark rectangle in a feed of bright ones is
# scrolled past - the same lesson pipeline/grade.py learned for the video itself.
SCENE = ("the monitor's broken screen blazing bright turquoise and red, cracked glass, "
         "glass splinters and glitch shards flying out into the room, a bedroom at night "
         "flooded with bright turquoise and red light from the screen and warm light from a "
         "lamp, high key, bright and well exposed, no deep shadows, vivid saturated colours, "
         "cinematic, photorealistic, 35mm film still")

# The face must be seen. "Staring at the monitor" alone let FLUX turn the person
# toward the screen and away from us - a back of a head sells nothing. Three
# quarters toward the camera keeps the eyes on the monitor and the fear on show.
FACE = ("terrified and screaming, eyes wide open, mouth wide open, the whole face clearly "
        "visible and brightly lit, turned three-quarters toward the camera, head and "
        "shoulders large in the left third of the frame, close up, eyes fixed on the right "
        "of the frame, never looking into the lens, never seen from behind, never with the "
        "back to the camera")


def base_prompts(seed, n=CANDIDATES):
    """One prompt per candidate: the whole scene in a single generation."""
    rng = random.Random(seed)
    who = rng.sample(PEOPLE, min(n, len(PEOPLE)))
    return [
        f"{p}, recoiling from a computer monitor on the right of the frame, {FACE}; "
        f"{CREATURES[(seed + i) % len(CREATURES)]}; "
        f"{SCENE}, {CLEAN}, {CREATURE_CLEAN}, {PERSON_CLEAN}"
        for i, p in enumerate(who)]


# When one generation will not hold both halves, each is drawn on its own and
# joined by compose_pair. The person is asked for with the right of the frame
# empty, so there is room for the monitor to be put there.
def person_prompts(seed, n=CANDIDATES):
    rng = random.Random(seed + 1)
    who = rng.sample(PEOPLE, min(n, len(PEOPLE)))
    return [
        f"{p}, recoiling, {FACE}, the right half of the frame an empty wall, a bedroom at "
        f"night flooded with bright turquoise and red light from the right, high key, bright "
        f"and well exposed, vivid saturated colours, cinematic, photorealistic, "
        f"35mm film still, {CLEAN}, {PERSON_CLEAN}"
        for p in who]


def creature_prompts(seed, n=CANDIDATES):
    return [
        f"a computer monitor on a desk, seen straight on, filling the frame; "
        f"{CREATURES[(seed + i) % len(CREATURES)]}; {SCENE}, no person, nobody else in the "
        f"room, {CLEAN}, {CREATURE_CLEAN}"
        for i in range(n)]


# --- framing ----------------------------------------------------------------------

def _fill(im):
    """Resize and centre-crop a generated frame to the thumbnail's 1280x720."""
    im = im.convert("RGB")
    s_ = max(W / im.width, H / im.height)
    im = im.resize((round(im.width * s_), round(im.height * s_)), Image.LANCZOS)
    x, y = (im.width - W) // 2, (im.height - H) // 2
    return im.crop((x, y, x + W, y + H))


# --- brightness ---------------------------------------------------------------------

# A cover sits among bright ones, so it is held above the video's own grade
# (pipeline/grade.py aims frames at 100). Not a filter for its own sake: the first
# v3 covers were mostly night, and a dark thumbnail reads as an empty one.
THUMB_MEAN = 118            # mean luma, 0-255, the cover is lifted toward
DARK_FLOOR = 40             # a pixel under this is near-black
DARK_SHARE_MAX = 0.22       # and no more than this share of the cover may be


def brightness(im):
    a = np.asarray(im.convert("L").resize((320, 180)), np.float32)
    return {"mean": round(float(a.mean()), 1),
            "dark_share": round(float((a < DARK_FLOOR).mean()), 3)}


def bright_enough(im):
    b = brightness(im)
    return b["mean"] >= THUMB_MEAN - 10 and b["dark_share"] <= DARK_SHARE_MAX, b


def brighten(im, target=THUMB_MEAN):
    """Lift a dark cover toward `target` with a gamma curve: the shadows open, the
    highlights stay where they were, nothing clips. A cover already bright enough
    is left alone. The lift is capped so a black frame does not turn to grey fog."""
    b = brightness(im)["mean"] / 255.0
    if b >= target / 255.0 or b <= 0.02:
        return im
    gamma = max(0.55, min(1.0, np.log(target / 255.0) / np.log(b)))
    a = np.asarray(im.convert("RGB"), np.float32) / 255.0
    return Image.fromarray(np.clip((a ** gamma) * 255.0, 0, 255).astype(np.uint8))


# --- words, the viewfinder, the checks --------------------------------------------

def _words(text):
    """Three words at most, or none at all. The reference cover sells the picture.
    A number keeps its thousands separators: "80,000 VIDEOS" is two words, not
    "80" and "000" and a lost noun."""
    return re.findall(r"\d{1,3}(?:,\d{3})+|[\w'?!.-]+", (text or "").upper())[:3]


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


def compose(base_path, words, seed, face=None):
    """One generated frame, framed. Returns (image, text size, word box)."""
    from pipeline import grade
    base = brighten(grade.grade(_fill(Image.open(base_path))))
    box, size = draw_words(base, words, face)
    return viewfinder(base, seed), size, box


def compose_pair(person_path, creature_path, words, seed, face=None, share=0.56):
    """The two-shot form: the monitor frame set into the right of the person frame.

    The seam is feathered rather than cut, and the monitor's light is blurred back
    across the whole frame, because the giveaway in a joined thumbnail is never the
    edge - it is two halves lit by different lamps.
    Returns (image, text size, word box)."""
    from pipeline import grade
    person = grade.grade(_fill(Image.open(person_path)))
    bw = int(W * share)
    cre = Image.open(creature_path).convert("RGB")
    s_ = max(bw / cre.width, H / cre.height)
    cre = cre.resize((round(cre.width * s_), round(cre.height * s_)), Image.LANCZOS)
    cx, cy = (cre.width - bw) // 2, (cre.height - H) // 2
    cre = grade.grade(cre.crop((cx, cy, cx + bw, cy + H)))

    layer = Image.new("RGB", (W, H), (0, 0, 0))
    layer.paste(cre, (W - bw, 0))
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rectangle((W - bw + 80, -1, W + 1, H + 1), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(48))
    out = person.copy()
    out.paste(layer, (0, 0), mask)
    # the screen lights the room both halves are standing in
    glow = layer.filter(ImageFilter.GaussianBlur(70))
    out = Image.fromarray(np.clip(np.asarray(out, np.int16)
                                  + np.asarray(glow, np.int16) // 5, 0, 255).astype(np.uint8))
    out = brighten(out)
    box, size = draw_words(out, words, face)
    return viewfinder(out, seed), size, box


def choose(bases, verdicts, words, out_dir, seed, pairs=None, small_judge=None):
    """Compose every candidate, check each one, return the best.

    `bases` are whole-scene frames. `pairs` is an optional list of
    (person, creature) frames composed with compose_pair and judged the same way,
    for the episodes one generation could not hold.

    `small_judge`, when given, is shown every finished cover at 336x189 - the size a
    viewer actually meets it at - and says whether the frightened face and the thing
    coming out of the screen can still be made out there. Words and contrast are
    measured here; whether an arm still reads as an arm takes eyes.

    A candidate passes when the base was approved, its words read small, the badge
    corner is clear, it is bright enough, and (with a judge) it reads at search size.
    Each report row carries "passes_all", and the winner's row carries "winner".

    Returns (path of the winner, report). Falls back to the best-scored candidate
    when none passes everything, and says so - a thumbnail is never left out, but a
    fallback never carries passes_all, so nothing downstream mistakes it for a pass."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report, paths = [], []
    jobs = [("single", b, None) for b in bases]
    jobs += [("pair", p, c) for p, c in (pairs or [])]
    for k, (kind, a, b) in enumerate(jobs):
        v = verdicts.get(Path(a), {})
        if kind == "single":
            im, size, box = compose(a, words, seed + k, v.get("face"))
        else:
            im, size, box = compose_pair(a, b, words, seed + k, v.get("face"))
        path = out_dir / f"thumb_{k}.jpg"
        im.save(path, quality=92)
        im.resize(SMALL, Image.LANCZOS).save(out_dir / f"thumb_{k}_small.jpg", quality=90)
        readable, m = readable_small(im, size, box)
        bright, b_ = bright_enough(im)
        row = {"file": path.name, "kind": kind,
               "ok": bool(v.get("ok", False)) and v.get("defect") != "unchecked",
               "defect": v.get("defect"), "note": v.get("note", ""),
               "readable": readable, "badge_corner_clear": safe_clear(box),
               "bright": bright, **b_, **m, "score": v.get("score", 0)}
        report.append(row)
        paths.append(path)

    if small_judge:
        small = small_judge([out_dir / f"thumb_{k}_small.jpg" for k in range(len(paths))])
        for k, row in enumerate(report):
            sv = small.get(out_dir / f"thumb_{k}_small.jpg", {})
            row["small_ok"] = bool(sv.get("ok", False))
            row["small_note"] = sv.get("note", "no verdict")
    for row in report:
        row["passes_all"] = bool(row["ok"] and row["readable"] and row["badge_corner_clear"]
                                 and row["bright"] and row.get("small_ok", small_judge is None))

    passing = [(r["score"], k) for k, r in enumerate(report) if r["passes_all"]]
    if passing:
        k = max(passing)[1]
    else:
        k = max(range(len(report)), key=lambda i: (report[i]["ok"], report[i]["readable"],
                                                   report[i]["bright"], report[i]["score"]))
        log.warning("no candidate passed every check; using %s", report[k]["file"])
    report[k]["winner"] = True
    final = out_dir / "thumbnail.jpg"
    Image.open(paths[k]).save(final, quality=92, optimize=True)
    if final.stat().st_size > 2_000_000:
        Image.open(paths[k]).save(final, quality=80, optimize=True)
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
