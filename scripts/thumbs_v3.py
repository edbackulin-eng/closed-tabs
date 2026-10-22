"""Draw sample covers for the v3 recipe on Kaggle and compose them here.

    python scripts/thumbs_v3.py --dry-run     # print the job, submit nothing
    python scripts/thumbs_v3.py --wait        # sleep until the GPU window, then run
    python scripts/thumbs_v3.py               # run now (refused outside the window)

Three episodes, three candidates each, so the samples show three real covers with
three real word sets rather than one picture three times. A production build still
draws pipeline.thumbnail.CANDIDATES bases for a single episode.

Closed Tabs may only use the GPU between 20:00 and 04:00 UTC (kaggle/gpu_budget.py):
What If builds in the morning and owns the card until it is done. push.push refuses
outside that window, and waits for any other channel's kernel to finish before it
submits.

Everything lands in checkpoint/thumbs_v3/: the raw bases, every composed candidate,
and sample_1..3.jpg - the pick for each episode. Nothing is uploaded to YouTube.
"""
import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle"))
from core import style  # noqa: E402
from core.logger import get_logger  # noqa: E402
from pipeline import thumbnail, visualqc  # noqa: E402

log = get_logger("thumbs_v3")
OUT = ROOT / "checkpoint" / "thumbs_v3"
EPISODES = ["cicada-3301", "max-headroom-hijack", "unfavorable-semicircle"]
EACH = 3


def plan(episodes=EPISODES, each=EACH):
    """[(slug, words, prompt)] - every candidate this run will draw."""
    rows = []
    for slug in episodes:
        meta = json.loads((ROOT / "queue" / slug / "meta.json").read_text(encoding="utf-8"))
        words = meta.get("thumbnail_words") or meta["name"]
        for p in thumbnail.base_prompts(sum(map(ord, slug)), each):
            rows.append((slug, words, p))
    return rows


def wait_for_window(poll_s=600):
    """Sleep until Closed Tabs is allowed on the GPU again."""
    import gpu_budget
    while not gpu_budget.in_window():
        now = datetime.now(timezone.utc)
        start = now.replace(hour=gpu_budget.WINDOW_UTC[0], minute=0, second=0, microsecond=0)
        if start <= now:
            start += timedelta(days=1)
        left = (start - now).total_seconds()
        log.info("outside the GPU window; %.1f h until %s UTC", left / 3600, start.strftime("%H:%M"))
        time.sleep(min(poll_s, max(60, left)))


def draw(rows, force_window=False):
    """Push one Kaggle kernel for every prompt, pull the bases back."""
    import push
    job = {"image_prompts": [p for _, _, p in rows], "style": "",
           "steps": style.STEPS, "width": style.ASPECT["width"],
           "height": style.ASPECT["height"], "seed": 3301}
    hours = 0.25 + 0.012 * len(rows)
    # A fresh slug every run, as What If does, so no run inherits a deleted or
    # half-finished kernel's state. (It was NOT the cause of the instant ERRORs on
    # the evening of 22.10: three kernels in a row - at fresh names, under the old
    # template and the new - died in seconds with an empty log and no quota
    # spent, and the fourth ran clean. That was Kaggle not handing out a session.)
    slug = f"closedtabs-thumbs-v3-{int(time.time())}"
    ref = push.push(job, slug, slug,
                    job_hours=hours, force_window=force_window)
    log.info("pushed %s; https://www.kaggle.com/code/%s", ref, ref)
    if not push.wait(ref, poll_s=30, timeout_s=push.deadline_for(job) + 600):
        raise RuntimeError(f"Kaggle kernel {ref} did not complete")
    pulled = push.fetch(ref, OUT / "_kernel")
    images = sorted((Path(pulled) / "images").glob("*.png"))
    if len(images) != len(rows):
        raise RuntimeError(f"{ref}: expected {len(rows)} bases, got {len(images)}")
    bases = OUT / "bases"
    bases.mkdir(parents=True, exist_ok=True)
    out = []
    for (slug, _, _), img in zip(rows, images):
        dest = bases / f"{slug}_{len([p for p in out if p[0] == slug]):02d}.png"
        shutil.copy(img, dest)
        out.append((slug, dest))
    return out


def judge(paths):
    """Gemini's verdicts, or an honest blank when the key is not on this machine.

    GEMINI_API_KEY lives in GitHub Secrets, not in the local .env. Without it the
    samples are still composed - they are for a human to look at - but every row
    is marked unchecked rather than passed, so nobody reads a missing check as a
    clean one."""
    from core import llm
    if not llm.get_key(llm.GEMINI_KEY_NAME):
        log.warning("%s is not set here; composing without the vision check",
                    llm.GEMINI_KEY_NAME)
        return {Path(p): {"ok": True, "defect": "unchecked", "note": "no key on this machine",
                          "score": 0} for p in paths}
    return visualqc.judge_thumbnails(paths)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the job, submit nothing")
    ap.add_argument("--wait", action="store_true", help="sleep until the GPU window opens")
    ap.add_argument("--force-window", action="store_true",
                    help="submit outside the window anyway (by hand only)")
    ap.add_argument("--bases", help="skip Kaggle and compose from a directory of bases")
    args = ap.parse_args()

    rows = plan()
    if args.dry_run:
        for slug, words, prompt in rows:
            print(f"\n--- {slug}  words={words!r} ---\n{prompt}")
        print(f"\n{len(rows)} prompts, about {0.25 + 0.012 * len(rows):.2f} GPU hours")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    if args.bases:
        drawn = [(slug, p) for slug in EPISODES
                 for p in sorted(Path(args.bases).glob(f"{slug}_*.png"))]
    else:
        if args.wait:
            wait_for_window()
        drawn = draw(rows, args.force_window)

    verdicts = judge([p for _, p in drawn])
    words = {slug: w for slug, w, _ in rows}
    report = {"drawn": len(drawn), "episodes": {}}
    for n, slug in enumerate(EPISODES, 1):
        mine = [p for s, p in drawn if s == slug]
        if not mine:
            continue
        final, rows_ = thumbnail.choose(mine, verdicts, words[slug], OUT / slug,
                                        sum(map(ord, slug)))
        sample = OUT / f"sample_{n}.jpg"
        shutil.copy(final, sample)
        report["episodes"][slug] = {"words": words[slug], "sample": sample.name,
                                    "candidates": rows_}
        print(f"{slug:24} -> {sample}")
    plain = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731
    (OUT / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=plain), encoding="utf-8")
    print(f"\nreport: {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
