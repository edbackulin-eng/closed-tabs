"""Generate, inspect and repair the stills an edit plan needs.

    python automation/stills.py queue/cicada-3301/build/hook_plan.json [--force-window]

1. FLUX.1-schnell on Kaggle, 1344x768, one kernel for every still in the plan,
   inside the Closed Tabs GPU window and the weekly budget (kaggle/gpu_budget.py).
2. Every still is checked: Gemini vision (lettering, watermarks, faces or eyes on
   the camera, hands and bodies, anything graphic, borders), a dark-edge test, and
   a perceptual hash against every other still for duplicates.
3. A failed still is generated once more with a new seed. If it fails again its
   prompt is swapped for a safe one and generated a last time; if even that fails,
   another approved still that is not used within ten shots takes its place.
4. Each annotate shot gets its red circle around a real object: Gemini names and
   boxes one, the pixels inside the box are checked for an actual object
   (visualqc.real_detail), and a shot with no verified detail loses its circle.

Stills land in <build>/stills/NNN.png, numbered by shot.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "kaggle"))
from core import style  # noqa: E402
from core.logger import get_logger  # noqa: E402
from pipeline import editplan, motion, visualqc  # noqa: E402

log = get_logger("stills")
NEEDS = {"flux", "annotate", "notification", "cursor"}
REPEAT_WINDOW = 10


def _generate(prompts, seed, dest, label, force_window, tag=""):
    """prompts: {shot: full prompt}. Writes dest/NNN.png for each shot."""
    import push
    order = sorted(prompts)
    job = {"image_prompts": [prompts[k] for k in order], "style": "",
           "steps": style.STEPS, "width": style.ASPECT["width"],
           "height": style.ASPECT["height"], "seed": seed}
    hours = 0.25 + 0.012 * len(order)          # load + ~40 s a still, generous
    # One kernel name per build, so a hook and a full episode built the same
    # evening never overwrite each other's kernel between push and fetch.
    slug = f"closedtabs-stills-{tag}-{label}" if tag else f"closedtabs-stills-{label}"
    ref = push.push(job, slug, slug, job_hours=hours, force_window=force_window)
    if not push.wait(ref, poll_s=30, timeout_s=3 * 3600):
        raise RuntimeError(f"Kaggle kernel {ref} did not complete")
    got = push.fetch(ref, dest / f"_{label}")
    images = sorted((dest / f"_{label}" / "images").glob("*.png"))
    if len(images) != len(order):
        raise RuntimeError(f"{ref}: expected {len(order)} stills, got {len(images)}")
    for k, img in zip(order, images):
        shutil.copy(img, dest / f"{k:03d}.png")
    return got


def _inspect(paths):
    """{path: verdict}, with the code-side checks folded in."""
    verdicts = visualqc.inspect(paths)
    for p in paths:
        ok, where = motion.check_edges(p)
        if not ok:
            verdicts[p] = {**verdicts.get(p, {}), "ok": False, "defect": "border",
                           "note": f"dark band at {where}"}
    for a, b in motion.duplicates(paths):
        later = Path(max(a, b))
        verdicts[later] = {**verdicts.get(later, {}), "ok": False, "defect": "duplicate",
                           "note": f"same picture as {Path(min(a, b)).name}"}
    return verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--force-window", action="store_true",
                    help="allow a run outside the GPU window (by hand only)")
    ap.add_argument("--thumbs", type=int, default=0,
                    help="also draw this many thumbnail bases in the same kernel")
    args = ap.parse_args()
    plan_path = Path(args.plan)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    dest = plan_path.parent / "stills"
    dest.mkdir(parents=True, exist_ok=True)
    shots = {s["move"]: s for s in plan["shots"]}
    prompts = {int(k): v for k, v in plan["prompts"].items()}

    # Thumbnail bases ride in the same kernel as the stills: one model load, not
    # two. They are numbered from 900 so they never collide with a shot.
    slug = plan_path.parent.parent.name
    import zlib
    tag = ("h" if plan_path.parent.name == "build" else "f") + f"{zlib.crc32(slug.encode()) % 10000:04d}"
    if args.thumbs:
        from pipeline import thumbnail
        seed = sum(map(ord, slug))
        for n, p in enumerate(thumbnail.base_prompts(seed, args.thumbs)):
            prompts.setdefault(900 + n, p)
    todo = {k: v for k, v in prompts.items() if not (dest / f"{k:03d}.png").exists()}
    if todo:
        _generate(todo, 3301, dest, "a", args.force_window, tag)
    thumbs = {k: prompts.pop(k) for k in [k for k in prompts if k >= 900]}
    if thumbs:
        tdir = plan_path.parent / "thumb_bases"
        tdir.mkdir(exist_ok=True)
        for k in thumbs:
            if (dest / f"{k:03d}.png").exists():
                shutil.move(str(dest / f"{k:03d}.png"), tdir / f"{k:03d}.png")

    report = {"rounds": []}
    for round_no, (seed, label) in enumerate(((4401, "b"), (5501, "c")), 1):
        paths = [dest / f"{k:03d}.png" for k in sorted(prompts)]
        verdicts = _inspect(paths)
        # A still nobody could look at (quota, a 503) is re-inspected next round,
        # not regenerated: drawing it again proves nothing and spends GPU time.
        unchecked = [p for p, v in verdicts.items() if v.get("defect") == "unchecked"]
        bad = {int(p.stem): v for p, v in verdicts.items()
               if not v["ok"] and v.get("defect") != "unchecked"}
        report.setdefault("unchecked", {})[f"round{round_no}"] = [p.name for p in unchecked]
        report["rounds"].append({"checked": len(paths),
                                 "failed": {f"{k:03d}": f"{v['defect']}: {v['note']}"
                                            for k, v in bad.items()}})
        log.info("round %d: %d of %d stills failed", round_no, len(bad), len(paths))
        for k, v in verdicts.items():
            s = shots.get(int(Path(k).stem))
            if s and s["type"] == "annotate" and "detail" in v:
                s["detail"] = v["detail"]
                if v["detail"]:
                    s["circle"] = v["detail"]["circle"]
        if not bad:
            break
        retry = {}
        for k in bad:
            if round_no == 2:                   # second failure: change the prompt
                safe = editplan.SAFE_STILLS[k % len(editplan.SAFE_STILLS)]
                shots[k]["prompt"] = safe
                retry[k] = style.scene_prompt(safe)
            else:
                retry[k] = prompts[k]
            (dest / f"{k:03d}.png").unlink(missing_ok=True)
        prompts.update(retry)
        _generate(retry, seed, dest, label, args.force_window, tag)
    else:
        # Still failing after the safe prompt: borrow an approved still that is not
        # on screen within REPEAT_WINDOW shots of this one.
        paths = [dest / f"{k:03d}.png" for k in sorted(prompts)]
        verdicts = _inspect(paths)
        good = [int(p.stem) for p, v in verdicts.items() if v["ok"]]
        for p, v in verdicts.items():
            if v["ok"]:
                continue
            k = int(p.stem)
            far = [g for g in good if abs(g - k) > REPEAT_WINDOW] or good
            if far:
                donor = max(far, key=lambda g: abs(g - k))
                shutil.copy(dest / f"{donor:03d}.png", p)
                report.setdefault("borrowed", {})[f"{k:03d}"] = f"{donor:03d}"
                log.warning("still %03d failed three times; using %03d", k, donor)

    # A circle goes only around a detail that was found and checked. An annotate
    # shot whose still has none becomes a plain moving still: no circle on air.
    for s in shots.values():
        if s["type"] == "annotate" and not s.get("detail"):
            s["type"] = "flux"
            s.pop("circle", None)
            report.setdefault("circle_dropped", []).append(s["move"])
    plan["prompts"] = {str(k): v for k, v in prompts.items()}
    plan_path.write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    (plan_path.parent / "stills_report.json").write_text(json.dumps(report, indent=1),
                                                         encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
