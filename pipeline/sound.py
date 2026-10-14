"""Sound: effects synthesised in code, and the final mix with its checks.

Effects are generated, not downloaded, so there is nothing to license: key and
mouse clicks, a whoosh, a digital glitch, a riser into each reveal and a low
impact on it.

The mix, in this order:
  1. the voice alone, two-pass loudnorm;
  2. the music bed set 19 LU under the voice (inside the 18-20 asked for) and
     ducked by about 6 dB while the voice speaks, so it is still heard in pauses;
  3. the effects bus;
  4. the sum, two-pass loudnorm to -14 LUFS with a -1 dBTP ceiling;
  5. a measurement. Anything outside the limits is corrected and measured again,
     and a mix that still fails raises - it is never handed to the uploader.
"""
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

from core.logger import get_logger

log = get_logger("sound")

SR = 48000
TARGET_LUFS = -14.0
TRUE_PEAK = -1.0
LUFS_TOL = 0.5
VOICE_LUFS = -16.0
MUSIC_UNDER_VOICE_LU = 19.0
# sidechaincompress settings that measured about 6 dB of gain reduction on
# speech in What If's tests (threshold 0.05, ratio 3 gave 6.7-7.8 dB there);
# ratio 2.5 sits a little lower, inside the 6-8 dB ceiling.
DUCK = "threshold=0.05:ratio=2.5:level_sc=2:attack=20:release=420"

_rng = np.random.default_rng(7)


# --- effects ---------------------------------------------------------------------

def _env(n, attack, decay):
    t = np.arange(n) / SR
    return np.minimum(1.0, t / max(attack, 1e-4)) * np.exp(-t / max(decay, 1e-4))


def _bandnoise(n, lo, hi):
    spec = np.fft.rfft(_rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / SR)
    spec[(f < lo) | (f > hi)] = 0
    x = np.fft.irfft(spec, n)
    return x / (np.abs(x).max() + 1e-9)


def key():
    n = int(SR * 0.05)
    hi = 2500 + _rng.uniform(-600, 600)
    return 0.5 * _bandnoise(n, hi * 0.6, hi * 2.2) * _env(n, 0.0005, 0.008)


def click():
    n = int(SR * 0.08)
    a = _bandnoise(n, 1500, 6000) * _env(n, 0.0003, 0.006)
    b = np.roll(_bandnoise(n, 900, 3500) * _env(n, 0.0003, 0.01), int(SR * 0.03))
    return 0.7 * a + 0.5 * b


def pop():
    n = int(SR * 0.12)
    t = np.arange(n) / SR
    f = 600 + 900 * np.exp(-t / 0.02)
    return 0.45 * np.sin(2 * np.pi * np.cumsum(f) / SR) * _env(n, 0.002, 0.04)


def tick():
    n = int(SR * 0.025)
    return 0.3 * _bandnoise(n, 3000, 9000) * _env(n, 0.0003, 0.004)


def whoosh(seconds=0.55):
    n = int(SR * seconds)
    t = np.linspace(0, 1, n)
    x = _bandnoise(n, 300, 7000)
    # a sweep: lowpass-like shaping by mixing a smoothed copy, swelling and falling
    smooth = np.convolve(x, np.ones(40) / 40, mode="same")
    mix = smooth * (1 - t) + x * t
    return 0.55 * mix * np.sin(np.pi * t) ** 1.5


def glitch(seconds=0.22):
    n = int(SR * seconds)
    x = _rng.standard_normal(n)
    hold = 60
    x = np.repeat(x[::hold], hold)[:n]             # sample-and-hold: digital crunch
    x = np.round(x * 3) / 3                         # bit-crush
    gate = (np.arange(n) // int(SR * 0.018)) % 2    # stutter
    return 0.35 * x * gate * _env(n, 0.001, seconds * 0.6)


def riser(seconds=1.8):
    n = int(SR * seconds)
    t = np.linspace(0, 1, n)
    f = 180 * (8 ** t)
    tone = np.sin(2 * np.pi * np.cumsum(f) / SR)
    noise = _bandnoise(n, 800, 9000)
    return 0.32 * (0.6 * tone + 0.5 * noise * t) * t ** 2


def impact(seconds=1.6):
    n = int(SR * seconds)
    t = np.arange(n) / SR
    f = 52 + 70 * np.exp(-t / 0.08)
    body = np.sin(2 * np.pi * np.cumsum(f) / SR) * np.exp(-t / 0.55)
    hit = _bandnoise(n, 60, 2500) * np.exp(-t / 0.05)
    return 0.9 * body + 0.35 * hit


MAKERS = {"key": (key, 0.30), "click": (click, 0.55), "pop": (pop, 0.45),
          "tick": (tick, 0.35), "whoosh": (whoosh, 0.40), "glitch": (glitch, 0.45),
          "riser": (riser, 0.50), "impact": (impact, 0.85)}


def effects_track(cues, seconds):
    """One mono track with every cue placed at its time."""
    out = np.zeros(int(SR * (seconds + 2)), np.float32)
    last_key = -1.0
    for c in sorted(cues, key=lambda c: c["t"]):
        kind = c["kind"]
        if kind == "key":
            if c["t"] - last_key < 0.045:
                continue                            # cap the typing rate
            last_key = c["t"]
        maker, gain = MAKERS[kind]
        s = maker() * gain
        i = int(SR * c["t"])
        j = min(len(out), i + len(s))
        if 0 <= i < len(out):
            out[i:j] += s[: j - i]
    return out[: int(SR * seconds)]


# --- loudness --------------------------------------------------------------------

def _run(args):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1500:])
    return r.stderr


def _measure(path, target=TARGET_LUFS, tp=TRUE_PEAK):
    err = _run(["-i", str(path), "-af",
                f"loudnorm=I={target}:TP={tp}:LRA=11:print_format=json", "-f", "null", "-"])
    a, b = err.rfind("{"), err.rfind("}")
    return json.loads(err[a:b + 1])


def loudnorm2(src, dest, target, tp):
    """Two-pass loudnorm to an exact integrated loudness."""
    m = _measure(src, target, tp)
    _run(["-y", "-i", str(src), "-af",
          f"loudnorm=I={target}:TP={tp}:LRA=11:measured_I={m['input_i']}:"
          f"measured_TP={m['input_tp']}:measured_LRA={m['input_lra']}:"
          f"measured_thresh={m['input_thresh']}:offset={m['target_offset']}:linear=true,"
          f"aresample={SR}", "-ar", str(SR), "-ac", "2", str(dest)])
    return dest


def measure(path):
    """Integrated loudness and true peak of a finished file (EBU R128)."""
    err = _run(["-i", str(path), "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"])
    import re
    i = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", err)
    p = re.findall(r"Peak:\s*(-?[\d.]+)\s*dBFS", err)
    return float(i[-1]), float(p[-1])


def _lufs(path):
    return float(_measure(path, VOICE_LUFS, -1.5)["input_i"])


# --- the mix -----------------------------------------------------------------------

def mix(voice_wav, music_path, cues, seconds, out_path, work):
    work = Path(work)
    voice = loudnorm2(voice_wav, work / "voice_norm.wav", VOICE_LUFS, -1.5)

    fx = effects_track(cues, seconds)
    sf.write(work / "fx.wav", np.stack([fx, fx], 1), SR)

    inputs = ["-i", str(voice), "-i", str(work / "fx.wav")]
    if music_path:
        gain = (VOICE_LUFS - MUSIC_UNDER_VOICE_LU) - _lufs(music_path)
        inputs += ["-stream_loop", "-1", "-i", str(music_path)]
        graph = (f"[2:a]aresample={SR},aformat=channel_layouts=stereo,volume={gain:.2f}dB,"
                 f"atrim=0:{seconds:.3f}[bed];"
                 f"[0:a]asplit=2[v][key];[bed][key]sidechaincompress={DUCK}[ducked];"
                 f"[1:a]volume=-6dB[fx];"
                 f"[v][ducked][fx]amix=inputs=3:duration=first:normalize=0[pre]")
        log.info("music bed gain %.1f dB (bed %.0f LU under the voice)", gain, MUSIC_UNDER_VOICE_LU)
    else:
        graph = "[1:a]volume=-6dB[fx];[0:a][fx]amix=inputs=2:duration=first:normalize=0[pre]"
    _run(["-y", *inputs, "-filter_complex", graph, "-map", "[pre]", "-t", f"{seconds:.3f}",
          "-ar", str(SR), "-ac", "2", str(work / "premix.wav")])

    loudnorm2(work / "premix.wav", out_path, TARGET_LUFS, TRUE_PEAK - 0.3)
    return verify(out_path, work)


def verify(path, work):
    """Measure; correct once if outside the limits; refuse if still outside."""
    lufs, peak = measure(path)
    log.info("mix: %.1f LUFS, true peak %.1f dBTP", lufs, peak)
    if abs(lufs - TARGET_LUFS) <= LUFS_TOL and peak <= TRUE_PEAK:
        return {"lufs": lufs, "true_peak": peak, "corrected": False}
    fixed = Path(work) / "mix_fixed.wav"
    gain = TARGET_LUFS - lufs
    _run(["-y", "-i", str(path), "-af",
          f"volume={gain:.2f}dB,alimiter=limit={10 ** ((TRUE_PEAK - 0.4) / 20):.4f}:"
          f"level=disabled:attack=2:release=60", "-ar", str(SR), str(fixed)])
    lufs2, peak2 = measure(fixed)
    log.info("corrected: %.1f LUFS, true peak %.1f dBTP", lufs2, peak2)
    if abs(lufs2 - TARGET_LUFS) > LUFS_TOL or peak2 > TRUE_PEAK:
        raise RuntimeError(f"mix out of limits after correction: {lufs2:.1f} LUFS, "
                           f"{peak2:.1f} dBTP; not uploading")
    Path(fixed).replace(path)
    return {"lufs": lufs2, "true_peak": peak2, "corrected": True}
