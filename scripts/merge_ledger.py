"""Merge this job's GPU ledger entries onto the ledger on main.

    python scripts/merge_ledger.py OURS.json

Two jobs that both record kernels collide on state/gpu_ledger.json when one
commits after the other; a plain rebase then stops on the conflict and the
hours go unrecorded (18.10). The ledger is a list of kernel runs, so the merge
is a union by (ref, start).
"""
import json
import sys
from pathlib import Path

LEDGER = Path(__file__).resolve().parent.parent / "state" / "gpu_ledger.json"
ours = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
theirs = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
seen = {(e["ref"], e["start"]) for e in theirs}
merged = theirs + [e for e in ours if (e["ref"], e["start"]) not in seen]
merged.sort(key=lambda e: e["start"])
LEDGER.write_text(json.dumps(merged, indent=1), encoding="utf-8")
print(f"ledger: {len(theirs)} on main + {len(merged) - len(theirs)} from this job")
