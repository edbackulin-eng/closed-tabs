"""Look at every frame before it reaches the render, with Gemini vision.

The technical QC measures duration, loudness and black frames. It cannot see
that a frame contains garbled AI text, a face staring down the lens, or a card
reading "?" - and all three shipped in the first cut of episode one precisely
because nothing was looking.

A frame that fails is regenerated once. If it fails again it is replaced rather
than argued with, because a second bad generation usually means the prompt is
the problem and the pipeline should not stall on it.

Runs on its own model so it cannot exhaust the fact-checker's daily quota.
"""
import base64
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from core import llm
from core.logger import get_logger

log = get_logger("visualqc")

# A third quota bucket, separate from the fact-check and the scene planner.
VISION_MODEL = "gemini-flash-lite-latest"
BATCH = 6                 # frames per request; each image costs tokens
# A 223-frame video is ~38 requests. Fired back to back, a third of them came
# back 429 and those frames went unchecked - which defeats the point of the
# check. Pacing them and retrying on rejection costs a couple of minutes.
MIN_INTERVAL_S = 2.0
RETRIES = 5
# 512 was too small to see garbled lettering on a phone screen - the defect
# that prompted this whole check went undetected at that size.
MAX_SIDE = 768

# Photographic frames and drawn cards fail in opposite directions: a photo must
# contain no lettering at all, while a card is nothing but lettering. Judging
# both with one rulebook flagged every clean card as a "text" defect.
PHOTO_SYSTEM = """You inspect photographic frames from a science documentary.

Report a defect ONLY when you can clearly see one of these:
- text: ANY lettering, words, captions, signage, UI, or a screen displaying
  characters - including text that is blurry, garbled or nonsensical. Garbled
  lettering is the most important defect to catch.
- face: a person facing the camera with their features LIT AND VISIBLE - you
  can make out eyes, nose and mouth and read an expression. Group shots count:
  three people standing and facing the lens is the same defect as one.
- anatomy: malformed hands, extra or missing limbs, distorted bodies

FINE, and never a defect: anyone seen from behind or turned away; a figure
backlit or silhouetted so the face reads as a dark shape, even in profile; a
face too distant or too dark to make out features. A screen glowing with NO
characters on it is FINE. Grain, darkness and blur are FINE.

The test is whether a viewer could describe the person's expression. If not, it
is a silhouette and it passes. Two calibration points, both real: a lit group of
three facing the lens IS a defect and was wrongly passed; a dark profile against
mist is NOT a defect and was wrongly flagged.
Reply with valid JSON only."""

ARCHIVE_SYSTEM = """You inspect real archive photographs from NASA and similar
agencies, used in a science documentary.

These are genuine photographs. Lettering that belongs to the real scene is FINE
and must NOT be reported: mission patches, "NASA" on hardware, "Canada" on the
ISS arm, instrument labels, map place names, colour-scale legends, timestamps
and frame identifiers burned in by the agency.

Report a defect ONLY when you see:
- branding: a watermark, logo or caption belonging to someone who is not the
  space agency - a broadcaster's bug, a stock-footage watermark, a documentary
  lower-third, a production company name. This is the one that matters: it is
  somebody else's trade mark in our frame.

Everything else about an archive photograph is acceptable.
Reply with valid JSON only."""

CARD_SYSTEM = """You inspect drawn infographic cards from a science documentary.

These cards are SUPPOSED to contain large text and numbers. Do not report that
as a defect.

Report a defect ONLY when:
- card: the card shows "?" as its main figure, a word is cut off mid-way
  (for example "8 billion people f" or "Inverse Cube (1/r^"), text runs past the
  edge of the frame, or the card is effectively empty.

Correct spelling, complete words and numbers are FINE however large.
Reply with valid JSON only."""

PROMPT = """For each image, in order, return:
{"results":[{"i":<index>,"ok":true|false,"defect":"text|face|anatomy|card|none",
"note":"<a few words, only when ok is false>"}]}"""


def _encode(path, max_side=MAX_SIDE):
    from PIL import Image
    import io
    with Image.open(path) as im:
        im = im.convert("RGB")
        scale = min(1.0, max_side / max(im.size))
        if scale < 1.0:
            im = im.resize((max(1, round(im.width * scale)),
                            max(1, round(im.height * scale))), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=82)
    return base64.b64encode(buf.getvalue()).decode("ascii")


_last_call = [0.0]


def _throttle():
    wait = MIN_INTERVAL_S - (time.monotonic() - _last_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_call[0] = time.monotonic()


def _inspect_batch(paths, model=VISION_MODEL, kind="photo"):
    key = llm.get_key("GEMINI_API_KEY")
    if not key:
        raise llm.LLMError("GEMINI_API_KEY is not set")
    parts = [{"text": PROMPT}]
    for i, p in enumerate(paths):
        parts.append({"text": f"image {i}:"})
        parts.append({"inline_data": {"mime_type": "image/jpeg",
                                      "data": _encode(p)}})
    payload = {
        "contents": [{"role": "user", "parts": parts}],
        "systemInstruction": {"parts": [
            {"text": {"card": CARD_SYSTEM,
                      "archive": ARCHIVE_SYSTEM}.get(kind, PHOTO_SYSTEM)}]},
        "generationConfig": {"maxOutputTokens": 2048, "temperature": 0.0,
                             "responseMimeType": "application/json"},
    }
    url = llm.GEMINI_URL.format(model=model)
    last = None
    for attempt in range(RETRIES):
        _throttle()
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode(), method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": key})
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8", "replace"))
            break
        except urllib.error.HTTPError as e:
            last = e
            # 503 is the service being unavailable, not us doing anything wrong,
            # and it has been the common failure rather than rate limiting.
            # Retrying it is the difference between a frame being inspected and
            # a frame shipping unlooked-at.
            if e.code not in (429, 500, 502, 503, 504):
                raise
            body = e.read().decode("utf-8", "replace") if e.code == 429 else ""
            if "PerDay" in body:          # a daily cap will not clear today
                raise
            backoff = 4 * (attempt + 1)
            log.info("vision %s, waiting %ds", e.code, backoff)
            time.sleep(backoff)
    else:
        raise last
    cand = (data.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "")
                   for p in cand.get("content", {}).get("parts", [])
                   if "text" in p)
    return llm.extract_json(text).get("results", [])


def inspect(paths, model=VISION_MODEL, kinds=None):
    """Returns {path: {"ok": bool, "defect": str, "note": str}}.

    `kinds` maps a path to "photo", "card" or "archive"; anything unlisted is a
    photo. Batches never mix kinds, because each has its own rulebook.

    Three rulebooks, not one, because the same pixel means different things by
    provenance. Lettering in a generated frame is always a defect; lettering in
    a NASA photograph is usually the label the agency put there; a card is
    nothing but lettering. Judging all three as "photo" flagged 36 frames of
    which 17 were perfectly good.
    """
    paths = [Path(p) for p in paths]
    kinds = {Path(k): v for k, v in (kinds or {}).items()}
    verdicts = {}
    groups = {"photo": [], "card": [], "archive": []}
    for p in paths:
        groups.setdefault(kinds.get(p, "photo"), []).append(p)

    done = 0
    # Each kind is batched separately: mixing them in one request would need two
    # rulebooks in one system prompt, and an earlier version that filtered a
    # mixed batch down to one kind silently dropped the rest.
    for kind, items in groups.items():
        for start in range(0, len(items), BATCH):
            batch = items[start:start + BATCH]
            try:
                results = _inspect_batch(batch, model, kind)
            except Exception as e:
                # Not "ok". A frame nobody looked at is unknown, not clean, and
                # recording it as clean is how a 503 turned into a shipped
                # defect. The caller decides what to do with unknowns.
                log.warning("vision batch (%s) failed (%s); %d frame(s) unchecked",
                            kind, e, len(batch))
                for p in batch:
                    verdicts[p] = {"ok": False, "defect": "unchecked",
                                   "note": str(e)[:80]}
                done += len(batch)
                continue
            by_i = {r.get("i"): r for r in results}
            for i, p in enumerate(batch):
                r = by_i.get(i, {})
                verdicts[p] = {"ok": bool(r.get("ok", True)),
                               "defect": (r.get("defect") or "none").lower(),
                               "note": r.get("note", "") or ""}
            done += len(batch)
            log.info("inspected %d/%d frames", done, len(paths))

    failed = [p for p, v in verdicts.items() if not v["ok"]]
    log.info("visual QC: %d frames, %d flagged", len(verdicts), len(failed))
    return verdicts


def write_report(verdicts, path):
    payload = {
        "frames": len(verdicts),
        "flagged": sum(1 for v in verdicts.values() if not v["ok"]),
        "unchecked": sum(1 for v in verdicts.values() if v["defect"] == "unchecked"),
        "details": [{"file": Path(p).name, **v}
                    for p, v in verdicts.items() if not v["ok"]],
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    return payload
