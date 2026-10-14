"""Build the one-minute hook of an episode in full edit: the checkpoint render.

    python automation/build_hook.py cicada-3301 [--stills DIR]

Expects queue/<slug>/build/ to hold hook.wav, hook_lines.json, hook_words.json and
hook_plan.json (automation/narrate_hook.py and plan_hook.py). Stills come from
--stills (NNN.png, numbered by shot) or from build/stills/.

Writes build/hook.mp4 and build/hook_report.json with the checks.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.logger import get_logger  # noqa: E402
from pipeline import assemble, motion, music, sound  # noqa: E402

log = get_logger("build-hook")

TRANSITION_SFX = {"vhs": "glitch", "rgb": "glitch", "fade": "whoosh"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--stills")
    args = ap.parse_args()
    b = ROOT / "queue" / args.slug / "build"
    work = b / "work"
    work.mkdir(parents=True, exist_ok=True)
    plan = json.loads((b / "hook_plan.json").read_text(encoding="utf-8"))
    shots = plan["shots"]
    stills_dir = Path(args.stills) if args.stills else b / "stills"
    frames = {int(p.stem): p for p in stills_dir.glob("*.png")}
    t0 = time.monotonic()

    report = {"checks": {}, "problems": []}
    edges = {}
    for k, p in frames.items():
        ok, where = motion.check_edges(p)
        edges[k] = where
        if not ok:
            report["problems"].append(f"still {k:03d}: dark band at {where}")
    dups = motion.duplicates(list(frames.values()))
    report["checks"]["duplicate_stills"] = dups
    if dups:
        report["problems"].append(f"{len(dups)} near-duplicate still pair(s)")

    paths, cues, cache = [], [], {}
    for s in shots:
        if s["type"] in ("flux", "annotate", "notification", "cursor") and s["move"] not in frames:
            raise RuntimeError(f"shot {s['move']} needs a still and has none in {stills_dir}")
        p, c = assemble.render_shot(s, frames, work, cache)
        paths.append(p)
        cues += c
        if s.get("transition"):
            cues.append({"t": s["start"], "kind": TRANSITION_SFX[s["transition"]]})
    log.info("shots rendered in %.0fs", time.monotonic() - t0)

    paths = assemble.apply_transitions(paths, shots, work)
    picture = assemble.concat(paths, work / "picture.mp4", work)

    acc = plan["accents"]
    cues += [{"t": t, "kind": "riser"} for t in acc.get("riser", [])]
    cues += [{"t": t, "kind": "impact"} for t in acc.get("impact", [])]

    import soundfile as sf
    seconds = sf.info(str(b / "hook.wav")).duration
    library = music.load_library()
    bed_plan = music.plan("", seconds, tracks=library, use_llm=False)
    bed = music.build_bed(bed_plan, work / "bed.wav")
    report["music"] = music.attribution(bed_plan)
    levels = sound.mix(b / "hook.wav", bed, cues, seconds, work / "mix.wav", work)
    report["checks"]["audio"] = levels

    out = b / "hook.mp4"
    assemble.finish(picture, work / "mix.wav", acc, out, work)
    from pipeline import qc
    meta = qc.probe(out)
    v = next(s for s in meta["streams"] if s["codec_type"] == "video")
    report["checks"].update({
        "duration_s": round(float(meta["format"]["duration"]), 2),
        "narration_s": round(seconds, 2),
        "resolution": f"{v['width']}x{v['height']}",
        "render_s": round(time.monotonic() - t0, 1),
        "shots": len(shots),
        "families": [s["family"] for s in shots],
        "types": [s["type"] for s in shots],
    })
    (b / "hook_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    print(json.dumps(report["checks"], indent=1))
    if report["problems"]:
        print("PROBLEMS:", report["problems"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
