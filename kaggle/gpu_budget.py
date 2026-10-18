"""When Closed Tabs may use the Kaggle GPU, and how much of the week is left.

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
    if not lines:
        return rows
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


def _whatif_builds(since):
    """How many What If build runs went past the first half hour since `since`;
    each of those used the GPU, under a kernel slug that is reused between runs."""
    import json as _j
    out = subprocess.run(["gh", "api", f"repos/{WHATIF_REPO}/actions/workflows/build-video.yml/runs",
                          "-X", "GET", "-f", f"created=>={since:%Y-%m-%d}", "-f", "per_page=50"],
                         capture_output=True, text=True, encoding="utf-8")
    try:
        runs = _j.loads(out.stdout).get("workflow_runs", [])
    except ValueError:
        return None
    n = 0
    for r in runs:
        try:
            a = datetime.fromisoformat(r["run_started_at"].replace("Z", "+00:00"))
            z = datetime.fromisoformat(r["updated_at"].replace("Z", "+00:00"))
        except (KeyError, TypeError, ValueError):
            continue
        if (z - a).total_seconds() > 1800:
            n += 1
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
    builds = _whatif_builds(since)
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
