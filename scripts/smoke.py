"""Smoke test: the whole chain on ten seconds of a real script, in a few minutes.

    python scripts/smoke.py

Run by .github/workflows/smoke.yml on every push to dev; only a passing run
moves the code to main, where the daily build runs it. It catches what broke
the 17.10 builds (a crash in planning) before any real build meets it.

  1. every module imports;
  2. the opening of a real queued script is narrated (Kokoro) and timed (Whisper);
  3. the edit is planned - on the code path and on a model-answer path fed
     deliberately bad answers (cut sentences), which must come back whole;
  4. placeholder stills stand in for FLUX, and the plan is rendered in two
     parts and merged the way the daily build does it;
  5. the result is checked: length against the narration, 1920x1080, loudness
     -14 LUFS and true peak, brightness over the floor, no black stretches,
     a thumbnail composes.
Exits non-zero, naming the failed check, when anything is wrong.
"""
import importlib
import json
import pkgutil
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle"))
W = ROOT / "work" / "smoke"
SECONDS = 10

failures = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        failures.append(f"{name}: {detail}")


def shared_gpu():
    """The shared GPU lock reads every channel's kernels the same way."""
    import gpu_shared
    ok = (gpu_shared.channel_of("u/whatif-main") == "whatif"
          and gpu_shared.channel_of("u/closedtabs-stills-a") == "closedtabs"
          and gpu_shared.channel_of("u/prehistoric-recon") is None
          and gpu_shared.channel_of("u/other") is None
          and gpu_shared.CHANNELS["whatif"] < gpu_shared.CHANNELS["closedtabs"])
    check("shared GPU lock", ok)


def imports():
    for pkg in ("core", "pipeline", "automation"):
        for m in pkgutil.iter_modules([str(ROOT / pkg)]):
            try:
                importlib.import_module(f"{pkg}.{m.name}")
            except Exception as e:                               # noqa: BLE001
                check(f"import {pkg}.{m.name}", False, repr(e)[:200])
    check("imports", not failures)


def narration():
    from automation.narrate_hook import opening, words_with_times
    from core import narrate
    from core.voice import SAMPLE_RATE
    q = json.loads((ROOT / "topics" / "queue.json").read_text(encoding="utf-8"))
    slug = next(t["slug"] for t in q["topics"] if (ROOT / "queue" / t["slug"] / "script.txt").exists())
    text = opening((ROOT / "queue" / slug / "script.txt").read_text(encoding="utf-8"), SECONDS)
    audio, lines = narrate.synthesize(narrate.load_kokoro(), text)
    wav = W / "narration.wav"
    sf.write(wav, audio, SAMPLE_RATE)
    words = words_with_times(wav)
    check("narration", len(audio) / SAMPLE_RATE > 4 and len(words) > 5,
          f"{slug}: {len(audio) / SAMPLE_RATE:.1f} s, {len(words)} words")
    return wav, lines, words, text


def planning(lines, words, total):
    from core import llm
    from pipeline import editplan
    plan = editplan.plan(lines, words, total, seed=1, providers=())
    check("plan (code path)", len(plan["shots"]) >= 2, f"{len(plan['shots'])} shots")

    real = llm.complete_json

    def bad_answers(prompt, **kw):
        beats = json.loads(prompt.split("BEATS:\n", 1)[1]) if "BEATS:" in prompt else []
        rows = []
        for b in beats:
            t = b["type"]
            if t == "censor":
                rows.append({"i": b["i"], "type": t, "heading": "file", "lines": ["Call us on a"],
                             "redact": ["Call"]})
            elif t == "quote":
                rows.append({"i": b["i"], "type": t, "text": "THEN, JULY 2015,"})
            elif t in ("browser",):
                rows.append({"i": b["i"], "type": t, "address": "a", "heading": "appears on",
                             "body": "If you"})
            else:
                rows.append({"i": b["i"], "type": t, "prompt": "an old book on a desk"})
        return {"beats": rows, "prompts": []}
    llm.complete_json = bad_answers
    try:
        plan2 = editplan.plan(lines, words, total, seed=2)
    finally:
        llm.complete_json = real
    cut = []
    for s in plan2["shots"]:
        texts = {"quote": [s.get("text", "")], "censor": s.get("lines", []),
                 "browser": [s.get("heading", "")]}.get(s["type"], [])
        cut += [t for t in texts if t and t.strip()[-1:] not in ".!?"]
    check("plan (model path, bad answers)", not cut, f"cut sentences left: {cut[:3]}")
    for s in plan["shots"]:                 # no stock downloads in the smoke test
        if s["type"] == "stock":
            s["type"] = "flux"
    return plan


def placeholder_stills(plan, dest):
    """Bright, varied stand-ins for FLUX frames: gradients with shapes."""
    dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    for s in plan["shots"]:
        y, x = np.mgrid[0:768, 0:1344]
        c1, c2 = rng.integers(40, 230, 3), rng.integers(40, 230, 3)
        t = (x / 1344)[..., None]
        a = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        im = Image.fromarray(a)
        d = ImageDraw.Draw(im)
        for _ in range(6):
            x0, y0 = rng.integers(0, 1200), rng.integers(0, 650)
            d.ellipse((x0, y0, x0 + rng.integers(40, 200), y0 + rng.integers(40, 200)),
                      fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
        im.save(dest / f"{s['move']:03d}.png")


def render(plan, wav, text):
    from pipeline import parts
    parts.PARTS = 2
    stills = W / "stills"
    placeholder_stills(plan, stills)
    t0 = time.monotonic()
    n = len(parts.split(plan["shots"], 2))
    for k in range(n):
        parts.render_part(json.loads(json.dumps(plan)), stills, k, W / f"part{k}", W / "parts")
    out = W / "video.mp4"
    report = parts.merge(plan, W / "parts", wav, out, W / "merge", script_text="")
    check("render + merge", out.exists(), f"{n} parts in {time.monotonic() - t0:.0f} s")
    return out, report


def checks(out, report, seconds):
    from pipeline import autofix, grade, qc, sound
    meta = qc.probe(out)
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    dur = float(meta["format"]["duration"])
    check("length", abs(dur - seconds) <= 0.5, f"{dur:.2f} s video, {seconds:.2f} s narration")
    check("resolution", (v["width"], v["height"]) == (1920, 1080), f"{v['width']}x{v['height']}")
    check("audio stream", any(s["codec_type"] == "audio" for s in meta["streams"]))
    lufs, peak = sound.measure(out)
    check("loudness", abs(lufs - sound.TARGET_LUFS) <= 1.0 and peak <= sound.TRUE_PEAK + 0.3,
          f"{lufs:.1f} LUFS, {peak:.1f} dBTP")
    b = grade.brightness(out, every_s=1.0)
    check("brightness", b["mean"] >= grade.VIDEO_MEAN_MIN, f"mean {b['mean']}")
    blacks = autofix.black_segments(out)
    check("no black stretches", not blacks, str(blacks))
    lv = report.get("checks", {}).get("music_levels")
    if lv:
        check("music under the voice", 10 <= lv["bed_under_voice_lu"] <= 16,
              f"{lv['bed_under_voice_lu']} LU, duck {lv['duck_db']} dB")


def thumbnail():
    from pipeline import thumbnail as th
    base, other = W / "thumb_base.png", W / "thumb_creature.png"
    im = Image.new("RGB", (1344, 768), (40, 60, 90))
    ImageDraw.Draw(im).rectangle((760, 160, 1240, 520), fill=(230, 232, 235))
    im.save(base)
    im2 = Image.new("RGB", (1344, 768), (20, 30, 50))
    ImageDraw.Draw(im2).rectangle((300, 140, 1050, 620), fill=(60, 220, 200))
    im2.save(other)

    out, size, box = th.compose(base, "STILL UNSOLVED", 1)
    check("thumbnail", out.size == (1280, 720), f"{out.size[0]}x{out.size[1]}")
    readable, m = th.readable_small(out, size, box)
    check("thumbnail reads at 336x189", readable, str(m))
    # the duration badge lands in the bottom right; nothing of ours may be under it
    check("thumbnail badge corner clear", th.safe_clear(box), str(box))

    # the two-shot form, for the episodes one generation could not hold
    pair, psize, pbox = th.compose_pair(base, other, "STILL UNSOLVED", 2)
    check("thumbnail (person + creature)", pair.size == (1280, 720)
          and th.readable_small(pair, psize, pbox)[0] and th.safe_clear(pbox))

    # eight prompts, each naming the creature, writing the film lookalike out, and
    # asking for the face to be seen rather than the back of a head
    prompts = th.base_prompts(3301)
    check("thumbnail prompts", len(prompts) == th.CANDIDATES
          and all("no long black hair" in p and "no white dress" in p
                  and "never seen from behind" in p for p in prompts),
          f"{len(prompts)} prompts")

    # a number keeps its thousands separator: "80,000" must not become "80" "000"
    check("thumbnail words", th._words("80,000 VIDEOS") == ["80,000", "VIDEOS"],
          str(th._words("80,000 VIDEOS")))

    # a dark base comes out lifted to a cover's brightness
    dark = W / "thumb_dark.png"
    Image.new("RGB", (1344, 768), (28, 32, 40)).save(dark)
    lit, _, _ = th.compose(dark, "", 3)
    check("thumbnail brightened", th.brightness(lit)["mean"] > 60, str(th.brightness(lit)))

    # choose(): every gate is recorded, exactly one winner, and a candidate whose
    # small-size verdict is missing never counts as a pass
    picked, rows = th.choose([base, other], {base: {"ok": True, "score": 7},
                                             other: {"ok": True, "score": 9}},
                             "80,000 VIDEOS", W / "choose", 4,
                             small_judge=lambda ps: {ps[0]: {"ok": True}})
    check("thumbnail choose", picked.exists() and sum(bool(r.get("winner")) for r in rows) == 1
          and rows[1]["small_ok"] is False and rows[1]["passes_all"] is False,
          str([(r["file"], r["passes_all"], r.get("winner", False)) for r in rows]))

    frames = sorted((W / "stills").glob("*.png"))[:5]
    if len(frames) >= 4:
        sheet = th.collage(frames, [f"Case {i}" for i in range(len(frames))],
                           "5 MOST DISTURBING", W / "roundup.jpg", 2)
        check("round-up collage", Image.open(sheet).size == (1280, 720), sheet.name)


def main():
    if W.exists():
        shutil.rmtree(W)
    W.mkdir(parents=True)
    imports()
    shared_gpu()
    if failures:
        return 1
    wav, lines, words, text = narration()
    seconds = sf.info(str(wav)).duration
    plan = planning(lines, words, seconds)
    out, report = render(plan, wav, text)
    checks(out, report, seconds)
    thumbnail()
    print("\nSMOKE", "PASSED" if not failures else f"FAILED: {failures}")
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
