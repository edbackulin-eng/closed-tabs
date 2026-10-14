"""One daily run: keep three videos scheduled ahead.

    python automation/daily.py [--no-upload]

1. Count the videos already scheduled and not yet public (state/episodes.json).
2. If fewer than BUFFER, take the next queued topic: write its script if it has
   none, build the episode, upload it private on the nearest free slot.
3. Record it in state/episodes.json at once.
4. Exit non-zero if, after all that, fewer than MIN_READY videos are scheduled -
   the run says so loudly rather than leaving a hole to be found on the channel.
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()
    rows = _state()
    have = ahead(rows)
    log.info("%d video(s) scheduled ahead, buffer %d", len(have), BUFFER)

    if len(have) < BUFFER:
        topic = next_topic(rows)
        if topic is None:
            log.error("the topic queue is empty")
        else:
            slug = topic["slug"]
            q = ROOT / "queue" / slug
            if not (q / "script.txt").exists():
                run([sys.executable, str(ROOT / "automation" / "write_script.py"), slug])
            run([sys.executable, str(ROOT / "automation" / "build_episode.py"), slug])
            if not args.no_upload:
                from pipeline import youtube
                out = q / "out"
                meta = json.loads((out / "metadata.json").read_text(encoding="utf-8"))
                slot = youtube.next_slot([r["publish_at"] for r in rows])
                vid = youtube.upload(out, meta["title"], meta["description"], meta["tags"], slot)
                youtube.attach(vid, out)
                rows.append({"slug": slug, "youtube_id": vid,
                             "publish_at": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
                             "title": meta["title"]})
                _save_state(rows)
                log.info("%s scheduled for %s as %s", slug, rows[-1]["publish_at"], vid)

    have = ahead(_state())
    print(f"scheduled ahead: {len(have)}")
    for r in sorted(have, key=lambda r: r["publish_at"]):
        print(f"  {r['publish_at']}  {r['youtube_id']}  {r['title']}")
    if len(have) < MIN_READY:
        log.error("only %d video(s) scheduled ahead; at least %d are needed", len(have), MIN_READY)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
