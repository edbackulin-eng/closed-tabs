"""When Closed Tabs may use the Kaggle GPU, and how much of the week is left.

Window. What If builds at 06:00 UTC on Mon, Tue, Thu, Fri and Sat, its job runs
up to 330 minutes, and GitHub has started scheduled runs of these repositories
up to 5 h 44 min late. The worst case therefore ends around 17:15 UTC. Closed
Tabs uses the GPU only between 20:00 and 04:00 UTC, which What If never
reaches, and checks that no kernel of either channel is running as well.

Budget. Both channels together stay under 25 GPU hours a week (Kaggle's own
free allowance is about 30). Closed Tabs measures its own jobs into
state/gpu_ledger.json. What If is not changed by this repository, so its use
is estimated: every What If kernel that ran in the last seven days is counted
at WHATIF_HOURS_PER_RUN, a deliberately pessimistic figure.
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


def whatif_runs(now=None):
    """What If kernels that ran in the last seven days, from Kaggle's listing."""
    since = (now or datetime.now(timezone.utc)) - timedelta(days=7)
    out = subprocess.run([sys.executable, "-m", "kaggle", "kernels", "list", "--mine",
                          "--page-size", "100", "--csv"], capture_output=True, text=True,
                         encoding="utf-8", errors="replace").stdout
    lines = out.splitlines()
    if not lines:
        return 0
    header = [h.strip() for h in lines[0].split(",")]
    try:
        i_ref, i_time = header.index("ref"), header.index("lastRunTime")
    except ValueError:
        return 0
    runs = 0
    for line in lines[1:]:
        cols = line.split(",")
        if len(cols) <= max(i_ref, i_time) or not cols[i_ref].split("/")[-1].startswith("whatif"):
            continue
        try:
            when = datetime.fromisoformat(cols[i_time].strip().replace("Z", "+00:00"))
        except ValueError:
            continue
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        if when >= since:
            runs += 1
    return runs


def check(job_hours=DEFAULT_JOB_H, now=None):
    """(ok, reason). Not ok when outside the window or the week would go over."""
    if not in_window(now):
        return False, (f"outside the Closed Tabs GPU window "
                       f"{WINDOW_UTC[0]:02d}:00-{WINDOW_UTC[1]:02d}:00 UTC")
    ct = closed_tabs_hours(now)
    wi = whatif_runs(now) * WHATIF_HOURS_PER_RUN
    if ct + wi + job_hours > WEEKLY_LIMIT_H:
        return False, (f"weekly GPU budget: Closed Tabs {ct:.1f} h + What If ~{wi:.1f} h "
                       f"+ this job {job_hours:.1f} h would pass {WEEKLY_LIMIT_H:.0f} h")
    return True, f"GPU week so far: Closed Tabs {ct:.1f} h, What If ~{wi:.1f} h"


if __name__ == "__main__":
    ok, why = check()
    print(("OK   " if ok else "NO   ") + why)
