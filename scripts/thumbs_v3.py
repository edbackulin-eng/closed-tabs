"""Draw v3 covers on Kaggle, check them, and ship one if it passes everything.

    python scripts/thumbs_v3.py --dry-run                 # print the job, submit nothing
    python scripts/thumbs_v3.py --wait                    # sleep until the GPU window, then run
    python scripts/thumbs_v3.py --wait --ship webdriver-torso

Run by .github/workflows/thumbs-v3.yml, where HF_TOKEN, the Gemini key and the
YouTube token come from GitHub Secrets; from a machine without them FLUX stops at
Hugging Face's gated repo and nothing can pass the vision checks.

The episode that goes out next gets the full set of candidates; the two after it
get three each, as samples of how the recipe reads on other stories.

Closed Tabs may only use the GPU between 20:00 and 04:00 UTC (kaggle/gpu_budget.py):
What If builds in the morning and owns the card until it is done. push.push refuses
outside that window, and waits for any other channel's kernel to finish before it
submits. --wait sleeps until the window opens, so the job can be started early.

Everything lands in checkpoint/thumbs_v3/: the raw bases, every composed candidate
and its 336x189 copy, sample_1..3.jpg - the pick for each episode in order - and
report.json. --ship SLUG sets that episode's pick as its YouTube thumbnail, but only
when the pick passed every check (thumbnail.choose's "passes_all") and the video has
not been published yet. Anything less and nothing on YouTube changes.
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
# (slug, candidates). The first is the cover that may ship; the rest are samples.
EPISODES = [("webdriver-torso", thumbnail.CANDIDATES), ("polybius", 3),
            ("markovian-parallax", 3)]


def plan(episodes=EPISODES):
    """[(slug, words, prompt)] - every candidate this run will draw."""
    rows = []
    for slug, n in episodes:
        meta = json.loads((ROOT / "queue" / slug / "meta.json").read_text(encoding="utf-8"))
        words = meta.get("thumbnail_words") or meta["name"]
        for p in thumbnail.base_prompts(sum(map(ord, slug)), n):
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

    Without the key the samples are still composed - they are for a human to look
    at - but every row is marked unchecked, and thumbnail.choose never lets an
    unchecked row pass, so nothing unexamined can be shipped."""
    from core import llm
    if not llm.get_key(llm.GEMINI_KEY_NAME):
        log.warning("%s is not set here; composing without the vision check",
                    llm.GEMINI_KEY_NAME)
        return {Path(p): {"ok": False, "defect": "unchecked", "note": "no key on this machine",
                          "score": 0} for p in paths}
    return visualqc.judge_thumbnails(paths)


def ship(slug, entry):
    """Set the pick as the episode's YouTube thumbnail, if and only if it passed every
    check and the video is still waiting to go out. Returns a record of what happened."""
    winner = next(r for r in entry["candidates"] if r.get("winner"))
    if not winner.get("passes_all"):
        failed = [k for k in ("ok", "readable", "badge_corner_clear", "bright", "small_ok")
                  if not winner.get(k, False)]
        return {"shipped": False, "why": f"the best candidate failed: {', '.join(failed)}",
                "winner": winner["file"]}
    episodes = json.loads((ROOT / "state" / "episodes.json").read_text(encoding="utf-8"))
    ep = next((e for e in episodes if e["slug"] == slug), None)
    if not ep or not ep.get("youtube_id"):
        return {"shipped": False, "why": f"{slug} has no YouTube video in state/episodes.json"}
    when = datetime.fromisoformat(ep["publish_at"].replace("Z", "+00:00"))
    if when <= datetime.now(timezone.utc):
        return {"shipped": False, "why": f"{slug} was already published at {ep['publish_at']}"}
    from pipeline import youtube
    youtube.attach(ep["youtube_id"], OUT / slug)          # sends OUT/slug/thumbnail.jpg
    return {"shipped": True, "youtube_id": ep["youtube_id"], "publish_at": ep["publish_at"],
            "winner": winner["file"], "at": datetime.now(timezone.utc).isoformat(timespec="seconds")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="print the job, submit nothing")
    ap.add_argument("--wait", action="store_true", help="sleep until the GPU window opens")
    ap.add_argument("--force-window", action="store_true",
                    help="submit outside the window anyway (by hand only)")
    ap.add_argument("--bases", help="skip Kaggle and compose from a directory of bases")
    ap.add_argument("--ship", default="",
                    help="episode slug whose pick goes to YouTube if it passes every check")
    args = ap.parse_args()

    rows = plan()
    if args.dry_run:
        for slug, words, prompt in rows:
            print(f"\n--- {slug}  words={words!r} ---\n{prompt}")
        print(f"\n{len(rows)} prompts, about {0.25 + 0.012 * len(rows):.2f} GPU hours")
        return 0

    OUT.mkdir(parents=True, exist_ok=True)
    if args.bases:
        drawn = [(slug, p) for slug, _ in EPISODES
                 for p in sorted(Path(args.bases).glob(f"{slug}_*.png"))]
    else:
        if args.wait:
            wait_for_window()
        drawn = draw(rows, args.force_window)

    verdicts = judge([p for _, p in drawn])
    words = {slug: w for slug, w, _ in rows}
    report = {"drawn": len(drawn), "episodes": {}}
    for n, (slug, _) in enumerate(EPISODES, 1):
        mine = [p for s, p in drawn if s == slug]
        if not mine:
            continue
        final, rows_ = thumbnail.choose(mine, verdicts, words[slug], OUT / slug,
                                        sum(map(ord, slug)), small_judge=visualqc.judge_small)
        sample = OUT / f"sample_{n}.jpg"
        shutil.copy(final, sample)
        winner = next(r for r in rows_ if r.get("winner"))
        report["episodes"][slug] = {"words": words[slug], "sample": sample.name,
                                    "passes_all": winner["passes_all"], "candidates": rows_}
        print(f"{slug:24} -> {sample}   passes all checks: {winner['passes_all']}")

    if args.ship:
        if args.ship not in report["episodes"]:
            report["ship"] = {"shipped": False, "why": f"no covers were made for {args.ship}"}
        else:
            report["ship"] = ship(args.ship, report["episodes"][args.ship])
        print("ship:", json.dumps(report["ship"]))
    plain = lambda o: o.item() if hasattr(o, "item") else str(o)  # noqa: E731
    (OUT / "report.json").write_text(
        json.dumps(report, indent=1, ensure_ascii=False, default=plain), encoding="utf-8")
    print(f"\nreport: {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
