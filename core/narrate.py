"""Closed Tabs narration: Kokoro am_michael, reading the delivery markup.

Pace is ~170 words a minute while speaking. Lines marked fast (hooks,
transitions) go up to 1.12; slow lines (reveals, key facts) go down to 0.95 and
are preceded by 0.6-0.9 s of silence. Names that Kokoro reads wrongly are
respelled phonetically first, from voice/pronunciations.json; subtitles and the
description keep the real spelling, because only the audio is rewritten.

Returns the audio and, for every line, where it starts and ends, so the edit
can cut to it without a forced alignment pass.
"""
import json
import re
from pathlib import Path

import numpy as np

from core import delivery
from core.logger import get_logger
from core.voice import SAMPLE_RATE, _trim_silence, peak_normalize

log = get_logger("narrate")

ROOT = Path(__file__).resolve().parent.parent
VOICE = "am_michael"
# am_michael measured 166 wpm at the nominal speeds on the reference fragment;
# this scales it to the channel's 170 and keeps every line inside 0.95-1.12.
VOICE_SCALE = 170 / 166
SPEED_MIN, SPEED_MAX = 0.95, 1.12
SLOW_LEAD = (0.6, 0.9)             # silence before a slow line, by its length
SENTENCE_TAIL_S = 0.20
QUESTION_EXTRA_S = 0.12
_SENT = re.compile(r"(?<=[.!?])\s+")


def _lexicon():
    data = json.loads((ROOT / "voice" / "pronunciations.json").read_text(encoding="utf-8"))
    return {k: v for k, v in data.items() if not k.startswith("_")}


def respell(text, lexicon=None):
    """The text as it should be spoken: known names rewritten phonetically."""
    lexicon = lexicon if lexicon is not None else _lexicon()
    for word in sorted(lexicon, key=len, reverse=True):
        text = re.sub(rf"(?<![\w-]){re.escape(word)}(?![\w-])", lexicon[word], text)
    return text


def _lead_for(line):
    """Longer reveals get the longer beat of silence before them."""
    n = len(line.text.split())
    lo, hi = SLOW_LEAD
    return round(lo + (hi - lo) * min(1.0, n / 10), 2)


def synthesize(kokoro, script, voice=VOICE):
    lexicon = _lexicon()
    chunks, timeline, t = [], [], 0.0
    for line in delivery.parse(script):
        pause = line.pause_before
        if line.kind == "slow":
            pause = max(pause, _lead_for(line))
        if pause:
            chunks.append(np.zeros(int(SAMPLE_RATE * pause), dtype=np.float32))
            t += pause
        speed = min(SPEED_MAX, max(SPEED_MIN, line.speed * VOICE_SCALE))
        start = t
        for sentence in [s for s in _SENT.split(line.text) if s.strip()]:
            audio, sr = kokoro.create(respell(sentence, lexicon), voice=voice,
                                      speed=speed, lang="en-us")
            audio = _trim_silence(np.asarray(audio, dtype=np.float32))
            chunks.append(audio)
            t += len(audio) / SAMPLE_RATE
            tail = SENTENCE_TAIL_S + (QUESTION_EXTRA_S if sentence.rstrip().endswith("?") else 0)
            chunks.append(np.zeros(int(SAMPLE_RATE * tail), dtype=np.float32))
            t += tail
        timeline.append({"text": line.text, "kind": line.kind, "start": round(start, 3),
                         "end": round(t, 3), "speed": round(speed, 3)})
    audio = peak_normalize(np.concatenate(chunks)) if chunks else np.zeros(1, np.float32)
    return audio, timeline


def load_kokoro():
    from kokoro_onnx import Kokoro
    models = ROOT / "assets" / "models"
    return Kokoro(str(models / "kokoro-v1.0.onnx"), str(models / "voices-v1.0.bin"))
