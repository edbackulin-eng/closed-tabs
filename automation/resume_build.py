"""Put a build that was cut off after its stills back together, without the GPU.

    python automation/resume_build.py cicada-3301 --jsons DIR

A full episode can run out of job time while rendering, after every still has
been drawn. Nothing heavy is lost: the Kaggle kernels keep their outputs, the
run's artifact keeps the build's JSON (plan, lines, words, stills report,
credits), the narration is deterministic and the stock files are re-fetched
from their sources. This rebuilds queue/<slug>/build_full/ from those, so
build_episode.py --skip-stills can render and package straight away.
"""
import argparse
import json
import shutil
import sys
import urllib.request
import zlib
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle"))
from core import narrate  # noqa: E402
from core.logger import get_logger  # noqa: E402
from core.voice import SAMPLE_RATE  # noqa: E402

log = get_logger("resume")
THUMBS = 8


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--jsons", required=True, help="the build_full JSON files from the artifact")
    args = ap.parse_args()
    q = ROOT / "queue" / args.slug
    b = q / "build_full"
    b.mkdir(parents=True, exist_ok=True)
    for f in Path(args.jsons).glob("*.json"):
        shutil.copy(f, b / f.name)
    plan = json.loads((b / "plan.json").read_text(encoding="utf-8"))
    report = json.loads((b / "stills_report.json").read_text(encoding="utf-8"))
    lines = json.loads((b / "lines.json").read_text(encoding="utf-8"))

    # 1. the narration: same script, same voice, same audio
    audio, _ = narrate.synthesize(narrate.load_kokoro(), (q / "script.txt").read_text(encoding="utf-8"))
    sf.write(b / "narration.wav", audio, SAMPLE_RATE)
    seconds = len(audio) / SAMPLE_RATE
    drift = abs(seconds - lines[-1]["end"])
    log.info("narration %.2f s, the plan was cut for %.2f s", seconds, lines[-1]["end"])
    if drift > 1.0:
        raise SystemExit(f"narration differs from the planned one by {drift:.1f} s; not resuming")

    # 2. the stills, round by round from the kernels that drew them
    import push
    tag = "f" + f"{zlib.crc32(args.slug.encode()) % 10000:04d}"
    stills, bases = b / "stills", b / "thumb_bases"
    stills.mkdir(exist_ok=True)
    bases.mkdir(exist_ok=True)
    keys_a = sorted({int(k) for k in plan["prompts"]} | set(range(900, 900 + THUMBS)))
    rounds = [("a", keys_a)]
    failed = [sorted(int(k) for k in r["failed"]) for r in report["rounds"]]
    if failed and failed[0]:
        rounds.append(("b", failed[0]))
    if len(failed) > 1 and failed[1]:
        rounds.append(("c", failed[1]))
    for label, keys in rounds:
        got = push.fetch(f"metberi/closedtabs-stills-{tag}-{label}", b / f"_k{label}")
        images = sorted((got / "images").glob("*.png"))
        if len(images) != len(keys):
            raise SystemExit(f"kernel {label}: {len(images)} images for {len(keys)} stills")
        for k, img in zip(keys, images):
            shutil.copy(img, (bases if k >= 900 else stills) / f"{k:03d}.png")
        log.info("round %s: %d stills", label, len(keys))
    for k, donor in report.get("borrowed", {}).items():
        shutil.copy(stills / f"{donor}.png", stills / f"{k}.png")

    # 3. the stock footage, fetched again from where it came from
    credits = {c["id"]: c for c in json.loads((b / "credits.json").read_text(encoding="utf-8"))}
    from pipeline import stock
    for s in plan["shots"]:
        a = s.get("asset")
        if s["type"] != "stock" or not a:
            continue
        p = Path(a["path"])
        p.parent.mkdir(parents=True, exist_ok=True)
        if not p.exists():
            url = credits.get(a["id"], {}).get("url")
            if not url:
                raise SystemExit(f"no source url for {a['id']}")
            p.write_bytes(stock._get(url, timeout=180))
    log.info("resume ready: %d stills, %d thumbnail bases", len(list(stills.glob("*.png"))),
             len(list(bases.glob("*.png"))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
