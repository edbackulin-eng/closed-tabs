"""When Closed Tabs may use the Kaggle GPU, and how much of the week is left.

Both channels see each other's kernels before they submit (kaggle/gpu_shared.py).
The prehistoric channel runs on Modal since 24.10 and is not counted here.

Window. What If builds at 06:00 UTC on Mon, Tue, Thu, Fri and Sat, its job runs
up to 330 minutes, and GitHub has started scheduled runs of these repositories
up to 5 h 44 min late. The worst case therefore ends around 17:15 UTC. Closed
Tabs uses the GPU only between 20:00 and 04:00 UTC, which What If never
reaches, and checks that no kernel of either channel is running as well.

Budget. Both channels together stay under 25 GPU hours a week (Kaggle's own
free allowance is about 30). Closed Tabs measures its own jobs into
state/gpu_ledger.json. What If is not changed by this repository, so its use
is measured from the logs of its Kaggle kernels (see whatif_hours).
"""
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEDGER = ROOT / "state" / "gpu_ledger.json"

WINDOW_UTC = (20, 4)                 # start hour, end hour (wraps past midnight)
WEEKLY_LIMIT_H = 25.0
WHATIF_HOURS_PER_RUN = 2.5
DEFAULT_JOB_H = 1.5                  # assumed for a job not yet measured


def in_window(now=None):
    hour = (now or datetime.now(timezone.utc)).hour
    start, end = WINDOW_UTC
    return hour >= start or hour < end


def _ledger():
    if LEDGER.exists():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return []


def record(ref, started, finished):
    rows = _ledger()
    rows.append({"ref": ref, "start": started.isoformat(), "end": finished.isoformat(),
                 "hours": round((finished - started).total_seconds() / 3600, 3)})
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(rows, indent=1), encoding="utf-8")


def closed_tabs_hours(now=None):
    since = (now or datetime.now(timezone.utc)) - timedelta(days=7)
    return sum(r["hours"] for r in _ledger()
               if datetime.fromisoformat(r["start"]) >= since)


WHATIF_CACHE = ROOT / "state" / "whatif_gpu.json"
WHATIF_REPO = "edbackulin-eng/whatif-channel"
MARGIN = 1.15                          # logs end slightly before the session is released


def _kernels():
    out = subprocess.run([sys.executable, "-X", "utf8", "-m", "kaggle", "kernels", "list", "--mine",
                          "--page-size", "100", "--csv"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace",
                         env={**__import__("os").environ, "PYTHONUTF8": "1",
                              "PYTHONIOENCODING": "utf-8"}).stdout
    rows = []
    lines = out.splitlines()
    # the CLI may print a version warning before the CSV: start at the header
    head = next((i for i, l in enumerate(lines) if l.startswith("ref,")), None)
    if head is None:
        return rows
    lines = lines[head:]
    header = [h.strip() for h in lines[0].split(",")]
    if "ref" not in header or "lastRunTime" not in header:
        return rows
    i_ref, i_time = header.index("ref"), header.index("lastRunTime")
    for line in lines[1:]:
        cols = line.split(",")
        if len(cols) <= max(i_ref, i_time):
            continue
        try:
            when = datetime.fromisoformat(cols[i_time].strip().replace("Z", "+00:00"))
        except ValueError:
            continue
        rows.append((cols[i_ref].strip(), when if when.tzinfo else when.replace(tzinfo=timezone.utc)))
    return rows


def _runtime_h(ref):
    """Measured runtime of a kernel's last run, from the timestamps in its log."""
    import json as _j
    import tempfile
    d = Path(tempfile.mkdtemp())
    subprocess.run([sys.executable, "-X", "utf8", "-m", "kaggle", "kernels", "output", ref, "-p",
                    str(d), "--file-pattern", r"\.log$", "-o"], capture_output=True,
                   env={**__import__("os").environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"})
    logs = list(d.glob("*.log"))
    if not logs or logs[0].stat().st_size == 0:
        return None
    try:
        entries = _j.loads(logs[0].read_text(encoding="utf-8"))
    except ValueError:
        return None
    times = [e.get("time", 0) for e in entries if isinstance(e, dict)]
    return max(times) / 3600 if times else None


PUSHED = b"successfully pushed"          # kaggle kernels push, in a What If build's log


def _pushed(run_id):
    """Did this What If build run push a kernel to Kaggle (its log says so)? None: unknown."""
    import io
    import zipfile
    out = subprocess.run(["gh", "api", f"repos/{WHATIF_REPO}/actions/runs/{run_id}/logs"], capture_output=True)
    if out.returncode != 0 or not out.stdout:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(out.stdout)) as z:
            return any(PUSHED in z.read(n) for n in z.namelist())
    except zipfile.BadZipFile:
        return None


_JOURNAL = {}
JOURNAL_URL = f"https://raw.githubusercontent.com/{WHATIF_REPO}/main/queue/journal.md"
_LINE = None


def _pushed_by_journal(run):
    """The same answer from What If's public journal (queue/journal.md), for when the run's
    log cannot be read - this repository's token may not read another repository's logs.
    Inside the run's time: "Built ..." -> a kernel ran; "Build FAILED ... kernels push ...
    failed" or "build skipped" only -> none ran; a failure of any other kind -> it may have."""
    import re
    import urllib.request
    global _LINE
    if "text" not in _JOURNAL:
        try:
            with urllib.request.urlopen(JOURNAL_URL, timeout=30) as r:
                _JOURNAL["text"] = r.read().decode("utf-8", "replace")
        except OSError:
            _JOURNAL["text"] = None
    if not _JOURNAL["text"]:
        return None
    _LINE = _LINE or re.compile(r"^- \*\*(\d{4}-\d\d-\d\d \d\d:\d\d) UTC\*\* - (.*)$", re.M)
    a = datetime.fromisoformat(run["run_started_at"].replace("Z", "+00:00")) - timedelta(minutes=1)
    z = datetime.fromisoformat(run["updated_at"].replace("Z", "+00:00")) + timedelta(minutes=1)
    seen = []
    for when, text in _LINE.findall(_JOURNAL["text"]):
        t = datetime.strptime(when, "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        if a <= t <= z:
            seen.append(text)
    if any(s.startswith("Built ") for s in seen):
        return True
    fails = [s for s in seen if s.startswith("Build FAILED")]
    if fails and all("kernels push" in s and "failed" in s for s in fails):
        return False
    if not fails and any("build skipped" in s or "Queue empty" in s for s in seen):
        return False
    return True if fails else None


def _whatif_builds(since, cache=None):
    """How many What If build runs since `since` ran a kernel on Kaggle (the build kernel's
    slug is reused between runs, so its log on Kaggle shows only the last one).

    Only runs whose log shows a kernel pushed count (05.11: the runs that waited 90 min for
    a "busy" GPU and then died on the kaggle CLI's import - no kernel ran - were counted as
    builds, What If came to ~14.8 h, and Closed Tabs drew 4 of 131 stills). A run still
    going, or one whose log cannot be read, counts if it went past the first half hour."""
    import json as _j
    out = subprocess.run(["gh", "api", f"repos/{WHATIF_REPO}/actions/workflows/build-video.yml/runs",
                          "-X", "GET", "-f", f"created=>={since:%Y-%m-%d}", "-f", "per_page=50"],
                         capture_output=True, text=True, encoding="utf-8")
    try:
        runs = _j.loads(out.stdout).get("workflow_runs", [])
    except ValueError:
        return None
    own = cache is None                  # whatif_hours passes its cache and writes it itself
    if own:
        cache = _j.loads(WHATIF_CACHE.read_text(encoding="utf-8")) if WHATIF_CACHE.exists() else {}
    n = 0
    for r in runs:
        try:
            a = datetime.fromisoformat(r["run_started_at"].replace("Z", "+00:00"))
            z = datetime.fromisoformat(r["updated_at"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            continue
        if (z - a).total_seconds() <= 1800:
            continue
        key = f"build-run:{r['id']}"
        if key not in cache and r.get("status") == "completed":
            p = _pushed(r["id"])
            if p is None:
                p = _pushed_by_journal(r)
            if p is not None:
                cache[key] = p
        n += 1 if cache.get(key, True) else 0
    if own:
        WHATIF_CACHE.parent.mkdir(parents=True, exist_ok=True)
        WHATIF_CACHE.write_text(_j.dumps(cache, indent=1), encoding="utf-8")
    return n


def whatif_hours(now=None):
    """What If's GPU hours in the last seven days, measured.

    The first version counted every What If kernel that had run in the week at a
    flat 2.5 hours. In a development week of eighteen mostly short kernels that
    came to 45 hours and blocked Closed Tabs outright. Now each kernel's last run
    is measured from its log (cached, so a log is fetched once), and the build
    kernel - whose slug is reused - is counted once per What If build run that week.
    """
    import json as _j
    now = now or datetime.now(timezone.utc)
    since = now - timedelta(days=7)
    cache = _j.loads(WHATIF_CACHE.read_text(encoding="utf-8")) if WHATIF_CACHE.exists() else {}
    total, main_h, seen_main = 0.0, None, False
    for ref, when in _kernels():
        if not ref.split("/")[-1].startswith("whatif") or when < since:
            continue
        key = f"{ref}@{when.isoformat()}"
        if key not in cache:
            h = _runtime_h(ref)
            cache[key] = round(h, 3) if h is not None else WHATIF_HOURS_PER_RUN
        total += cache[key]
        if ref.endswith(("whatif-main", "whatif-retry")):
            main_h = max(main_h or 0, cache[key])
            seen_main = True
    builds = _whatif_builds(since, cache)
    if seen_main and builds and main_h:
        total += main_h * max(0, builds - 1)          # earlier runs of the reused slug
    WHATIF_CACHE.parent.mkdir(parents=True, exist_ok=True)
    WHATIF_CACHE.write_text(_j.dumps(cache, indent=1), encoding="utf-8")
    return total * MARGIN


def check(job_hours=DEFAULT_JOB_H, now=None):
    """(ok, reason). Not ok when outside the window or the week would go over."""
    if not in_window(now):
        return False, (f"outside the Closed Tabs GPU window "
                       f"{WINDOW_UTC[0]:02d}:00-{WINDOW_UTC[1]:02d}:00 UTC")
    ct = closed_tabs_hours(now)
    wi = whatif_hours(now)
    if ct + wi + job_hours > WEEKLY_LIMIT_H:
        return False, (f"weekly GPU budget: Closed Tabs {ct:.1f} h + What If ~{wi:.1f} h "
                       f"+ this job {job_hours:.1f} h would pass {WEEKLY_LIMIT_H:.0f} h")
    return True, f"GPU week so far: Closed Tabs {ct:.1f} h, What If ~{wi:.1f} h"


# Kept back for What If's next build, so Closed Tabs never spends the hours it needs.
WHATIF_RESERVE_H = WHATIF_HOURS_PER_RUN


def remaining_h(now=None):
    """GPU hours Closed Tabs may still spend this week."""
    return WEEKLY_LIMIT_H - closed_tabs_hours(now) - whatif_hours(now) - WHATIF_RESERVE_H


if __name__ == "__main__":
    ok, why = check()
    print(("OK   " if ok else "NO   ") + why)
