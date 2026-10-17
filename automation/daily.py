"""One daily run: keep three videos scheduled ahead.

    python automation/daily.py [--no-upload]

1. Count the videos already scheduled and not yet public (state/episodes.json).
2. If fewer than BUFFER, take the next queued topic: write its script if it has
   none, build the episode, upload it private on the nearest free slot.
3. Record it in state/episodes.json at once.
4. Exit non-zero if this run added nothing while fewer than MIN_READY videos are
   scheduled - the run says so loudly rather than leaving a hole on the channel.
"""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402

log = get_logger("daily")
STATE = ROOT / "state" / "episodes.json"
QUEUE = ROOT / "topics" / "queue.json"
BUFFER = 3
MIN_READY = 2


def _state():
    return json.loads(STATE.read_text(encoding="utf-8")) if STATE.exists() else []


def _save_state(rows):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")


def ahead(rows, now=None):
    now = now or datetime.now(timezone.utc)
    return [r for r in rows
            if datetime.fromisoformat(r["publish_at"].replace("Z", "+00:00")) > now]


def next_topic(rows):
    done = {r["slug"] for r in rows}
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    for t in queue["topics"]:
        if t["slug"] not in done and t.get("status", "queued") == "queued":
            return t
    return None


def run(cmd):
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def schedule(slug, rows):
    """Upload a built episode (queue/<slug>/out/) private on the nearest free slot,
    record it in the state and mark the topic done. Returns the new row."""
    from pipeline import youtube
    out = ROOT / "queue" / slug / "out"
    meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
    slot = youtube.next_slot([r["publish_at"] for r in rows])
    vid = youtube.upload(out, meta["title"], meta["description"], meta["tags"], slot)
    youtube.attach(vid, out)
    rows.append({"slug": slug, "youtube_id": vid,
                 "publish_at": slot.strftime("%Y-%m-%dT%H:%M:%SZ"), "title": meta["title"]})
    _save_state(rows)
    # the topic is done: the next run takes the next one
    queue = json.loads(QUEUE.read_text(encoding="utf-8"))
    for t in queue["topics"]:
        if t["slug"] == slug:
            t["status"] = "published"
    QUEUE.write_text(json.dumps(queue, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info("%s scheduled for %s as %s", slug, rows[-1]["publish_at"], vid)
    return rows[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    ap.add_argument("--force-window", action="store_true",
                    help="by hand only: allow Kaggle outside the GPU window")
    ap.add_argument("--upload-only", metavar="SLUG",
                    help="schedule an episode that is already built in queue/SLUG/out")
    args = ap.parse_args()
    rows = _state()
    if args.upload_only:
        if any(r["slug"] == args.upload_only for r in rows):
            log.info("%s is already scheduled; nothing to do", args.upload_only)
        else:
            schedule(args.upload_only, rows)
        for r in sorted(ahead(_state()), key=lambda r: r["publish_at"]):
            print(f"  {r['publish_at']}  {r['youtube_id']}  {r['title']}")
        return 0
    have = ahead(rows)
    log.info("%d video(s) scheduled ahead, buffer %d", len(have), BUFFER)

    produced = False
    if len(have) < BUFFER:
        topic = next_topic(rows)
        if topic is None:
            log.error("the topic queue is empty")
        else:
            slug = topic["slug"]
            q = ROOT / "queue" / slug
            if not (q / "script.txt").exists():
                run([sys.executable, str(ROOT / "automation" / "write_script.py"), slug])
            run([sys.executable, str(ROOT / "automation" / "build_episode.py"), slug,
                 *(["--force-window"] if args.force_window else [])])
            if not args.no_upload:
                schedule(slug, rows)
                produced = True

    have = ahead(_state())
    print(f"scheduled ahead: {len(have)}")
    gh_out = __import__("os").environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"ahead={len(have)}\nbuffer={BUFFER}\n")
    for r in sorted(have, key=lambda r: r["publish_at"]):
        print(f"  {r['publish_at']}  {r['youtube_id']}  {r['title']}")
    # A run that added a video did its job even if the buffer is still filling;
    # the workflow starts the next build. A run that added nothing while the
    # buffer is short is a failure, and says so (retry, then an email).
    if len(have) < MIN_READY and not produced and not args.no_upload:
        log.error("only %d video(s) scheduled ahead; at least %d are needed", len(have), MIN_READY)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
