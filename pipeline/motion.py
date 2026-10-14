"""Ken Burns movement on a still, the way this channel does it.

scale + crop with eval=frame, never zoompan: zoompan rounds its offsets to
whole pixels at output size and walks the frame in visible steps, and on these
machines it ran at a twentieth of real time. Here the still is supersampled to
twice the output (3840x2160), scaled per frame, cropped, and only then brought
down to 1920x1080, so a one-pixel step at the working size is half a pixel on
screen. The zoom runs 1.00 to 1.10 on a smoothstep curve, so a shot starts and
ends at rest instead of lurching, and the direction alternates shot to shot.

A FLUX frame arrives at 1344x768 (1.75:1). It is filled to 16:9 by cropping a
sliver off the sides, never by letterboxing; check_edges() is the automatic
proof that no dark band reached the frame.
"""
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image

OUT_W, OUT_H, FPS = 1920, 1080, 30
SS = 2                                    # supersampling factor
WORK_W, WORK_H = OUT_W * SS, OUT_H * SS
ZOOM = 0.10                               # 1.00 -> 1.10

# Each move: (zoom in?, horizontal drift -1..1, vertical drift -1..1)
MOVES = [(True, 0.0, 0.0), (False, -0.6, 0.0), (True, 0.6, -0.3),
         (False, 0.0, 0.4), (True, -0.5, 0.3), (False, 0.5, -0.2)]


def prepare(src, dest):
    """Fill 16:9 at the working size. Returns the path of the prepared still."""
    im = Image.open(src).convert("RGB")
    scale = max(WORK_W / im.width, WORK_H / im.height)
    im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    x, y = (im.width - WORK_W) // 2, (im.height - WORK_H) // 2
    im.crop((x, y, x + WORK_W, y + WORK_H)).save(dest, quality=95)
    return dest


def _filter(duration, move):
    zoom_in, dx, dy = move
    e = f"(st(0,clip(t/{duration:.3f},0,1));ld(0)*ld(0)*(3-2*ld(0)))"
    z = f"(1+{ZOOM}*{e})" if zoom_in else f"(1+{ZOOM}-{ZOOM}*{e})"
    # Scaled size, kept even. The crop then picks the window; its centre drifts
    # by dx, dy across the spare margin as the shot runs.
    w = f"trunc({WORK_W}*{z}/2)*2"
    h = f"trunc({WORK_H}*{z}/2)*2"
    cx = f"(iw-{WORK_W})*(0.5+{dx / 2:.3f}*({e}-0.5))"
    cy = f"(ih-{WORK_H})*(0.5+{dy / 2:.3f}*({e}-0.5))"
    return (f"scale=w='{w}':h='{h}':eval=frame:flags=bicubic,"
            f"crop={WORK_W}:{WORK_H}:x='{cx}':y='{cy}',"
            f"scale={OUT_W}:{OUT_H}:flags=lanczos,fps={FPS},format=yuv420p")


def kenburns(still, out_path, duration, index=0):
    """Render one moving shot from a prepared 3840x2160 still."""
    move = MOVES[index % len(MOVES)]
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-loop", "1",
         "-framerate", str(FPS), "-t", f"{duration:.3f}", "-i", str(still),
         "-vf", _filter(duration, move), "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "18", "-r", str(FPS), str(out_path)], check=True)
    return out_path


# --- automatic checks ------------------------------------------------------

def check_edges(path_or_img, band=0.03, floor=0.18):
    """True when no edge of the frame is a dark band.

    Compares each edge strip with the frame's own median brightness, so a dark
    night scene is not mistaken for a letterbox: a band is flagged only when an
    edge is both very dark and much darker than the picture inside it.
    """
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    a = np.asarray(im.convert("L"), np.float32)
    h, w = a.shape
    bh, bw = max(2, int(h * band)), max(2, int(w * band))
    inner = np.median(a[bh:h - bh, bw:w - bw]) + 1e-3
    strips = {"top": a[:bh], "bottom": a[h - bh:], "left": a[:, :bw], "right": a[:, w - bw:]}
    problems = [name for name, s in strips.items()
                if s.mean() < 12 and s.mean() < inner * floor]
    return not problems, problems


def phash(path_or_img, size=32, keep=8):
    """64-bit perceptual hash (DCT of a 32x32 greyscale), as an int."""
    from scipy.fft import dct
    im = path_or_img if isinstance(path_or_img, Image.Image) else Image.open(path_or_img)
    a = np.asarray(im.convert("L").resize((size, size), Image.LANCZOS), np.float32)
    d = dct(dct(a, axis=0, norm="ortho"), axis=1, norm="ortho")[:keep, :keep]
    bits = (d > np.median(d[1:, 1:])).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a, b):
    return bin(a ^ b).count("1")


def duplicates(paths, threshold=10):
    """Pairs of images closer than `threshold` bits: the same picture twice."""
    hashes = [(p, phash(p)) for p in paths]
    out = []
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming(hashes[i][1], hashes[j][1]) < threshold:
                out.append((str(hashes[i][0]), str(hashes[j][0])))
    return out
