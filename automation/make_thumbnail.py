"""Judge the thumbnail bases, compose every candidate, keep the best.

    python automation/make_thumbnail.py cicada-3301
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import thumbnail, visualqc  # noqa: E402


def main(slug):
    q = ROOT / "queue" / slug
    b = q / "build"
    meta = json.loads((q / "meta.json").read_text(encoding="utf-8"))
    bases = sorted((b / "thumb_bases").glob("*.png"))
    if not bases:
        raise SystemExit("no thumbnail bases in build/thumb_bases")
    verdicts = visualqc.judge_thumbnails(bases)
    script = (q / "script.txt").read_text(encoding="utf-8")
    quote = next((l.lstrip("<> ").strip() for l in script.splitlines()
                  if l.strip() and not l.startswith("[")), meta["name"])
    words = meta.get("thumbnail_words") or meta["name"]
    final, report = thumbnail.choose(bases, verdicts, words, meta["name"], quote[:70],
                                     b / "thumbs", sum(map(ord, slug)))
    out = {"winner": final.name, "candidates": report,
           "verdicts": {Path(k).name: v for k, v in verdicts.items()}}
    (b / "thumbnail_report.json").write_text(json.dumps(out, indent=1, ensure_ascii=False),
                                             encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
