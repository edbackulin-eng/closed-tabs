"""Judge the thumbnail bases, compose every candidate, keep the best.

    python automation/make_thumbnail.py cicada-3301
    python automation/make_thumbnail.py cicada-3301 build_full
    python automation/make_thumbnail.py --collage week-42

The round-up form reads queue/<slug>/roundup.json:

    {"words": "5 MOST DISTURBING",
     "items": [{"frame": "build/stills/004.png", "caption": "Cicada 3301"}, ...]}

with four to six items; each frame is a path inside the episode directory.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import thumbnail, visualqc  # noqa: E402


def main(slug, build="build"):
    q = ROOT / "queue" / slug
    b = q / build
    meta = json.loads((q / "meta.json").read_text(encoding="utf-8"))
    bases = sorted((b / "thumb_bases").glob("*.png"))
    if not bases:
        raise SystemExit("no thumbnail bases in build/thumb_bases")
    verdicts = visualqc.judge_thumbnails(bases)
    words = meta.get("thumbnail_words") or meta["name"]
    final, report = thumbnail.choose(bases, verdicts, words, b / "thumbs", sum(map(ord, slug)))
    out = {"winner": final.name, "candidates": report,
           "verdicts": {Path(k).name: v for k, v in verdicts.items()}}
    # numpy scalars (a numpy bool from the readability test) are not JSON: the
    # first full episode failed here, after its thumbnail had been made
    plain = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731
    (b / "thumbnail_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False, default=plain),
                                             encoding="utf-8")
    print(json.dumps(report, indent=1, default=plain))
    return 0


def roundup(slug, build="build"):
    """The weekly collage cover. No FLUX and no vision call: the frames are stills
    that were already inspected when their own episodes were built."""
    q = ROOT / "queue" / slug
    spec = json.loads((q / "roundup.json").read_text(encoding="utf-8"))
    items = spec["items"]
    missing = [i["frame"] for i in items if not (q / i["frame"]).exists()]
    if missing:
        raise SystemExit(f"round-up frames not found: {missing}")
    out = thumbnail.collage([q / i["frame"] for i in items],
                            [i.get("caption", "") for i in items],
                            spec.get("words", ""), q / build / "thumbs" / "thumbnail.jpg",
                            sum(map(ord, slug)))
    print(out)
    return 0


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--collage"]
    sys.exit((roundup if "--collage" in sys.argv else main)(*args[:2]))
