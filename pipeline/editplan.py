"""The edit: which picture is on screen when, and what moves on it.

The rhythm is decided here, by code, so it holds whatever a model answers:
  - a cut every 3-5 seconds, snapped to a gap between spoken words;
  - a new family of visual every 20-40 seconds: photo (generated stills, moving or
    annotated), screen (browser, chat, notification, cursor) and data (counter,
    timeline, redacted document) take turns, so the eye never settles;
  - a punch-in on every slow line, a riser into it and an impact on it, and a
    small shake on the last reveal.

The model is only asked to fill each beat with content of the family already
chosen for it, from the narration being spoken over it. Anything it returns that
does not fit the menu is replaced by a safe default for that family.
"""
import json
import random
import re

from core import llm
from core.logger import get_logger

log = get_logger("editplan")

CUT_MIN, CUT_MAX = 3.0, 5.0
BLOCK_MIN, BLOCK_MAX = 20.0, 40.0
FAMILIES = ("photo", "screen", "data")
TYPES = {
    "photo": ("flux", "annotate"),
    "screen": ("browser", "chat", "notification", "cursor"),
    "data": ("quote", "counter", "timeline", "censor"),
}
TRANSITIONS = ("vhs", "rgb", "fade")
PLAN_BATCH = 25

PLANNER_SYSTEM = """You plan the pictures for a documentary-style YouTube video about an
internet mystery. You are given beats: a few seconds of narration each, and the visual
TYPE already chosen for that beat. Fill in that type's fields. Return JSON only.

RULES FOR GENERATED STILLS (any "prompt" field)
- Describe a photograph in one sentence: a place, an object, light. Moody, at night,
  lit by a screen or a lamp.
- No real logos, no real websites or apps, no brand names, no readable text.
- Never ask for an object that carries writing: no calendars, newspapers, clippings,
  notes, letters, posters, signs, maps, books, documents, labels, or screens showing
  content. Screens are only glow. Writing is drawn separately by code.
- No identifiable people. At most one person, seen from behind or in silhouette,
  never looking at the camera. No children.
- Nothing violent, no blood, no weapons.

RULES FOR DRAWN SCREENS AND DATA
- Text must come from the narration of that beat or be a neutral label. Do not invent
  facts, numbers or quotes the narration does not state.
- Use generic labels instead of real site names ("image board", "forum", "inbox").
- Keep it short: an address under 40 characters, a heading under 8 words, a body
  under 30 words, at most 3 chat messages of under 12 words each.

TYPES AND FIELDS
  flux:         {"prompt": "..."}
  annotate:     {"prompt": "...", "circle": [x, y, r]}    (x, y, r between 0 and 1)
  browser:      {"address": "...", "heading": "...", "body": "..."}
  chat:         {"header": "...", "messages": [["who", "text"], ...]}
  notification: {"prompt": "...", "app": "...", "text": "..."}
  cursor:       {"prompt": "..."}
  counter:      {"label": "...", "end": 123, "prefix": "", "suffix": ""}   (only for a number the narration says)
  timeline:     {"events": [["when", "what"], ...]}                      (2-4 events from the narration)
  censor:       {"heading": "...", "lines": ["...", "..."], "redact": ["word", ...]}
  quote:        {"text": "the 6-14 most striking words of this beat, exactly as spoken"}

Return: {"beats": [{"i": 0, "type": "...", ...fields}, ...]}
"""


# --- rhythm -----------------------------------------------------------------

def _gaps(words):
    """Moments between spoken words, where a cut does not land mid-syllable."""
    return [(a["end"] + b["start"]) / 2 for a, b in zip(words, words[1:])]


def cut_points(words, total, seed=0):
    rng = random.Random(seed)
    gaps = _gaps(words) or [total]
    cuts, t = [0.0], 0.0
    while total - t > CUT_MAX:
        want = t + rng.uniform(3.4, 4.6)
        near = [g for g in gaps if t + CUT_MIN <= g <= t + CUT_MAX]
        t = min(near, key=lambda g: abs(g - want)) if near else t + 4.0
        cuts.append(round(t, 3))
    if total - cuts[-1] < CUT_MIN and len(cuts) > 1:
        cuts.pop()
    cuts.append(round(total, 3))
    return cuts


def assign_families(cuts, seed=0):
    """Blocks of 20-40 s, families in turn. Every video opens on photographs: the
    first fifteen seconds sell the story, and a moving still does that better
    than a window of text."""
    rng = random.Random(seed + 1)
    # Photo every other block, screen and data taking turns between: about half
    # the running time is moving stills, a quarter each screens and data. Equal
    # thirds made a third of a 14-minute video text cards.
    order = ["photo", "screen", "photo", "data"]
    span = {"photo": (26, 36), "screen": (22, 32), "data": (20, 26)}
    fams, f, block_end = [], len(order) - 1, 0.0      # first increment lands on photo
    for a, b in zip(cuts, cuts[1:]):
        if a >= block_end:
            f = (f + 1) % len(order)
            block_end = a + rng.uniform(*span[order[f]])
        fams.append(order[f])
    return fams


def _spoken(words, a, b):
    return " ".join(w["word"] for w in words if a <= (w["start"] + w["end"]) / 2 < b).strip()


# --- content ------------------------------------------------------------------

def _whole_sentences(text):
    """Complete sentences only. A beat starts and ends wherever a cut fell, so its
    first and last pieces are usually halves of sentences ("...decryption. And the
    website was removed. Just") and are dropped."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    out = []
    for k, p in enumerate(parts):
        p = p.strip()
        if not p:
            continue
        starts = p[:1].isupper() or p[:1].isdigit()
        ends = p[-1:] in ".!?"
        if (k == 0 and not starts) or (k == len(parts) - 1 and not ends):
            continue
        out.append(p)
    return out


def _redactable(sentence):
    """Words worth a censor bar: names and numbers, never the first word."""
    words = sentence.split()
    return [w.strip(".,;:!?\"'") for k, w in enumerate(words)
            if k > 0 and (w[:1].isupper() or any(c.isdigit() for c in w))
            and len(w.strip(".,;:!?\"'")) > 1]


def _default(t, text, i, still=None):
    """A safe beat of type `t` when the model's answer cannot be used.

    Per type, not per family, so a bad answer does not undo the rotation that keeps
    two identical shots from following each other.
    """
    still = still or SAFE_STILLS[i % len(SAFE_STILLS)]
    words = text.split()
    if t == "flux":
        return {"type": t, "prompt": still}
    if t == "annotate":
        return {"type": t, "prompt": still, "circle": [0.5, 0.48, 0.2]}
    if t == "browser":
        return {"type": t, "address": "archive / thread", "heading": " ".join(words[:5]) or "Thread",
                "body": " ".join(words[5:30]) or " ".join(words)}
    if t == "chat":
        half = max(1, len(words) // 2)
        msgs = [["anon", " ".join(words[:half][:12])]]
        if words[half:]:
            msgs.append(["user", " ".join(words[half:][:12])])
        return {"type": t, "header": "Thread", "messages": msgs}
    if t == "notification":
        return {"type": t, "prompt": still, "app": "Inbox", "text": " ".join(words[:12]) or "New message"}
    if t == "cursor":
        return {"type": t, "prompt": still}
    if t in ("quote", "censor"):
        sents = _whole_sentences(text)
        if t == "censor":
            lines = sents[:3]
            redact = [w for s in lines for w in _redactable(s)][:3]
            if lines and redact:
                return {"type": t, "heading": "file", "lines": lines, "redact": redact}
            # nothing worth blacking out: a quote says more than random bars
        best = max((s for s in sents if len(s.split()) <= 14), key=lambda s: len(s.split()),
                   default=None)
        if not best:
            best = " ".join(words[:12]).rstrip(",;:") + "..." if words else "..."
        return {"type": "quote", "text": best}
    if t == "counter":
        n = re.search(r"\d[\d,]*", text)
        return {"type": t, "label": " ".join(words[:4]) or "count",
                "end": int(n.group(0).replace(",", "")) if n else 1}
    if t == "timeline":
        years = re.findall(r"\b(?:19|20)\d\d\b", text)
        if len(set(years)) >= 2:
            return {"type": t, "events": [[y, ""] for y in dict.fromkeys(years)][:4]}
        return _default("quote", text, i, still)
    return _default("quote", text, i, still)


_ALIASES = {"url": "address", "link": "address", "title": "heading", "headline": "heading",
            "text_body": "body", "content": "body", "description": "prompt",
            "image": "prompt", "image_prompt": "prompt", "msgs": "messages", "chat": "messages",
            "value": "end", "number": "end", "items": "events", "redacted": "redact"}


def _normalise(b):
    """Map the field names models reach for onto the menu's names."""
    out = dict(b)
    for k, v in b.items():
        if k in _ALIASES and _ALIASES[k] not in out:
            out[_ALIASES[k]] = v
    fields = out.pop("fields", None)
    if isinstance(fields, dict):
        out.update({k: v for k, v in fields.items() if k not in out})
    if out.get("type") == "browser" and "body" not in out and "text" in out:
        out["body"] = out["text"]
    if isinstance(out.get("messages"), list):
        msgs = []
        for m in out["messages"]:
            if isinstance(m, dict):
                m = [m.get("who") or m.get("user") or m.get("from") or "anon",
                     m.get("text") or m.get("message") or ""]
            if isinstance(m, (list, tuple)) and len(m) >= 2 and str(m[1]).strip():
                msgs.append([str(m[0]), str(m[1])])
        out["messages"] = msgs
    if isinstance(out.get("events"), list):
        ev = []
        for e in out["events"]:
            if isinstance(e, dict):
                e = [e.get("when") or e.get("date") or e.get("year") or "",
                     e.get("what") or e.get("text") or e.get("event") or ""]
            if isinstance(e, (list, tuple)) and len(e) >= 2:
                ev.append([str(e[0]), str(e[1])])
        out["events"] = ev
    return out


def _why_invalid(b, family):
    t = b.get("type")
    if t not in TYPES[family]:
        return f"type {t!r} not in {family}"
    need = {"flux": ["prompt"], "annotate": ["prompt", "circle"], "browser": ["address", "heading", "body"],
            "chat": ["messages"], "notification": ["prompt", "app", "text"], "cursor": ["prompt"],
            "counter": ["label", "end"], "timeline": ["events"], "censor": ["lines", "redact"],
            "quote": ["text"]}[t]
    missing = [k for k in need if k not in b or b[k] in ("", [], None)]
    return f"{t} missing {missing}" if missing else None


def _facts_hold(b, narration):
    """Dates and numbers drawn on screen must be ones the narration actually says.

    The planner, asked for a timeline, wrote "2014 puzzle begins", "2015 first phone
    number" - plausible and wrong; the puzzle began in 2012. So every digit-bearing
    label is checked against the words being spoken over that beat.
    """
    said = set(re.findall(r"\d[\d,.]*", narration.replace(",", "")))
    t = b.get("type")
    if t == "timeline":
        for when, _ in b.get("events", []):
            for num in re.findall(r"\d[\d,.]*", str(when).replace(",", "")):
                if num not in said:
                    return False
        return bool(b.get("events"))
    if t == "counter":
        return str(b.get("end")).replace(",", "") in said
    if t == "quote":
        spoken = set(re.findall(r"[a-z0-9']+", narration.lower()))
        q = re.findall(r"[a-z0-9']+", str(b.get("text", "")).lower())
        return bool(q) and sum(w in spoken for w in q) / len(q) >= 0.8
    for field in ("heading", "body", "text", "label"):
        for num in re.findall(r"\d[\d,.]*", str(b.get(field, "")).replace(",", "")):
            if num not in said:
                return False
    return True


def _valid(b, family):
    t = b.get("type")
    if t not in TYPES[family]:
        return False
    need = {"flux": ["prompt"], "annotate": ["prompt", "circle"], "browser": ["address", "heading", "body"],
            "chat": ["messages"], "notification": ["prompt", "app", "text"], "cursor": ["prompt"],
            "counter": ["label", "end"], "timeline": ["events"], "censor": ["lines", "redact"],
            "quote": ["text"]}[t]
    if any(k not in b or b[k] in ("", [], None) for k in need):
        return False
    if t == "counter":
        try:
            b["end"] = int(re.sub(r"[^0-9-]", "", str(b["end"])))
        except ValueError:
            return False
    return True


# Objects FLUX cannot draw without lettering on them. A prompt that asks for one
# is refused before it reaches the GPU: the visual check would only reject the
# frame afterwards, and the planner kept reaching for them (a torn calendar page
# with a date, newspaper clippings, a pinned map).
_TEXT_OBJECTS = re.compile(
    r"\b(calendar|newspaper|clipping|headline|post-?it|sticky note|note ?pad|notes?|letter|"
    r"envelope|sign|signs|signage|poster|billboard|label|map|maps|book cover|magazine|"
    r"document|page|pages|printout|receipt|ticket|license plate|keyboard keys|"
    r"screen showing|on the screen|on a screen|website|web page|logo|brand|"
    r"displaying|showing|notice|thread|post|posts|message|messages|notification|"
    r"text|words|writing|caption|subtitle|interface|keypad|dial pad|"
    r"forum|homepage|home page|browser|cursor|app|icon|icons)\b", re.I)

SAFE_STILLS = [
    "an empty desk at night lit by one old CRT monitor, dust in the air",
    "a dark bedroom, the glow of a laptop on a closed curtain",
    "a tangle of network cables in a dim server cupboard, blinking indicator lights",
    "an empty office corridor at night, one ceiling light flickering",
    "rain on a window at night, the blurred lights of a city beyond",
    "an old rotary telephone on a wooden table in a dark room, lamp light",
    "a shortwave radio on a shelf, its dial glowing amber in the dark",
    "a hand resting on a computer mouse beside a glowing monitor, seen from behind",
    "a payphone on an empty street corner at night under a sodium streetlight",
    "a row of server racks receding into darkness, small green lights",
    "a cold mug of coffee beside a keyboard in a dark room, screen glow on the wall",
    "a dusty attic with a single bare bulb and cardboard boxes in shadow",
    "a city skyline at night seen from a high window, one apartment light on",
    "an old television set glowing in an empty living room",
    "a long empty hallway with fluorescent tubes, one of them out",
    "a desk lamp pooling warm light on a dark wooden desk, nothing else on it",
    "a satellite dish on a rooftop at night against a deep blue sky",
    "a rain-soaked parking lot at night, one car under a flickering light",
    "a tall radio antenna mast at dusk with red warning lights",
    "a dark stairwell lit by a single green exit glow",
    "headphones resting on a desk beside a glowing monitor in a dark room",
    "a window at 3 a.m., blinds half closed, the street lamp outside",
    "an empty swivel chair in front of a glowing monitor, seen from behind",
    "a pile of old computer hard drives under a lamp in a dark workshop",
    "a basement with exposed pipes and a single glowing computer in the corner",
    "fog over an empty road at night, distant headlights",
    "a laptop closed on a bed, a faint light leaking from its edge",
    "a wall of old CRT monitors, all glowing blank blue in a dark room",
    "a quiet suburban street at night, one window lit upstairs",
    "a tangle of old cables and a modem with blinking lights on a carpet",
    "a hand holding a phone face down on a table, the edge glowing",
    "an empty internet cafe at night, rows of dark monitors, one still on",
]

REPAIR_SYSTEM = """You rewrite image prompts for a photo generator that cannot draw text.
Each prompt below was refused because it names something that carries writing (a
calendar, a page, a sign, a screen showing content, a website...). Rewrite each as a
moody photograph of a place or object with NO writing anywhere: screens are only
glow, nothing is printed, nothing is labelled. Keep the mood and the idea. One
sentence each. Return JSON only: {"prompts": [{"i": <index>, "prompt": "..."}]}"""


def assign_types(beats):
    """Rotate the types inside each family so no shot repeats the one before it."""
    turn = {f: 0 for f in FAMILIES}
    prev = None
    for b in beats:
        options = list(TYPES[b["family"]])
        if b["family"] == "data":
            if not re.search(r"\d", b["text"]):
                options = [t for t in options if t != "counter"]
            if len(re.findall(r"(?:19|20)\d\d", b["text"])) < 2:
                options = [t for t in options if t != "timeline"]
        if b["family"] == "photo":
            options = ["flux", "flux", "annotate"]      # mostly moving stills
        k = turn[b["family"]]
        t = options[k % len(options)]
        if t == prev and len(set(options)) > 1:
            k += 1
            t = options[k % len(options)]
        turn[b["family"]] = k + 1
        b["type"], prev = t, t
    return beats


def plan(lines, words, total, seed=0, providers=llm.PROVIDERS):
    cuts = cut_points(words, total, seed)
    fams = assign_families(cuts, seed)
    beats = [{"i": i, "start": a, "end": b, "family": fams[i], "text": _spoken(words, a, b)}
             for i, (a, b) in enumerate(zip(cuts, cuts[1:]))]
    assign_types(beats)
    # In batches: two hundred beats in one request came back truncated and every
    # one of them fell to a default. A batch that fails costs only its own beats.
    filled = {}
    for k in range(0, len(beats), PLAN_BATCH):
        part = beats[k:k + PLAN_BATCH]
        ask = json.dumps([{"i": b["i"], "type": b["type"], "narration": b["text"]} for b in part],
                         ensure_ascii=False)
        try:
            answer = llm.complete_json(f"BEATS:\n{ask}", system=PLANNER_SYSTEM, max_tokens=6000,
                                       temperature=0.6, providers=providers)
            rows = answer.get("beats", []) if isinstance(answer, dict) else answer
            filled.update({int(x.get("i", -1)): x for x in rows if isinstance(x, dict)})
        except Exception as e:                                   # noqa: BLE001
            log.warning("planner batch %d failed (%s); its beats get defaults", k // PLAN_BATCH, e)
    # Prompts that ask for lettering go back to the model once, to be rewritten
    # without it; only what is still refused after that falls to the safe pool.
    bad = {i: x["prompt"] for i, x in ((i, _normalise(x)) for i, x in filled.items())
           if x.get("prompt") and _TEXT_OBJECTS.search(x["prompt"])}
    for k in range(0, len(bad), PLAN_BATCH):
        part = dict(list(bad.items())[k:k + PLAN_BATCH])
        try:
            a = llm.complete_json("PROMPTS:\n" + json.dumps([{"i": i, "prompt": p} for i, p in part.items()]),
                                  system=REPAIR_SYSTEM, max_tokens=3000, temperature=0.5,
                                  providers=providers)
            for r in (a.get("prompts", []) if isinstance(a, dict) else a):
                i, p = int(r.get("i", -1)), str(r.get("prompt", ""))
                if i in filled and p and not _TEXT_OBJECTS.search(p):
                    filled[i]["prompt"] = p
        except Exception as e:                                   # noqa: BLE001
            log.warning("prompt repair failed (%s)", e)

    shots, used_defaults, refused = [], 0, 0
    # One pool for every substitute still, so no two shots draw the same one
    # until all of them are used; then a still nobody has used for a while.
    spare = list(SAFE_STILLS)
    used_at = {}

    def next_safe(i):
        if spare:
            p = spare.pop(0)
        else:
            p = min(SAFE_STILLS, key=lambda q: used_at.get(q, -999))
        used_at[p] = i
        return p
    for b in beats:
        got = _normalise(filled.get(b["i"], {}))
        got["type"] = b["type"]
        if got.get("prompt") and _TEXT_OBJECTS.search(got["prompt"]):
            log.info("beat %d: prompt asks for lettering (%s), replaced", b["i"],
                     _TEXT_OBJECTS.search(got["prompt"]).group(0))
            got["prompt"] = next_safe(b["i"])
            refused += 1
        if got.get("type") == "chat":
            # Nobody real speaks in these threads: handles are always faceless.
            got["messages"] = [["anon" if k % 2 == 0 else "user", m[1]]
                               for k, m in enumerate(got.get("messages") or [])][:3]
        if not _valid(got, b["family"]):
            log.info("beat %d: %s; default used", b["i"], _why_invalid(got, b["family"]))
            got = _default(b["type"], b["text"], b["i"], next_safe(b["i"]))
            used_defaults += 1
        elif not _facts_hold(got, b["text"]):
            log.info("beat %d: %s shows a number the narration does not say; default used",
                     b["i"], got["type"])
            got = _default(b["type"], b["text"], b["i"], next_safe(b["i"]))
            used_defaults += 1
        got.update({"start": b["start"], "end": b["end"], "family": b["family"],
                    "narration": b["text"]})
        shots.append(got)
    for k, s in enumerate(shots):
        s["transition"] = TRANSITIONS[k % 3] if k else None
        s["move"] = k
    log.info("%d shots, %d defaults, %d prompts refused for lettering; families: %s",
             len(shots), used_defaults, refused, " ".join(s["family"][0] for s in shots))
    return {"shots": shots, "accents": accents(lines, words)}


def accents(lines, words):
    """Punch-ins, risers, impacts, kinetic words and the final shake, from the delivery.

    Every slow line is a reveal: a riser runs into it, an impact lands on its first
    word, the frame punches in, and its strongest word or two appear on screen. The
    last slow line also gets a small shake.
    """
    out = {"punch": [], "riser": [], "impact": [], "kinetic": [], "shake": []}
    slow = [l for l in lines if l["kind"] == "slow"]
    for n, l in enumerate(slow):
        first = next((w for w in words if w["start"] >= l["start"] - 0.05), None)
        if not first:
            continue
        hit = first["start"]
        out["riser"].append(round(max(0.0, hit - 1.8), 3))
        out["impact"].append(round(hit, 3))
        out["punch"].append([round(hit, 3), round(min(l["end"], hit + 2.2), 3)])
        line_words = [w for w in words if l["start"] - 0.05 <= w["start"] < l["end"]]
        pick = _strong_words(_script_words(l, line_words))
        if pick:
            out["kinetic"].append({"words": [w["word"] for w in pick],
                                   "start": pick[0]["start"],
                                   "end": round(max(pick[-1]["end"] + 0.9, pick[0]["start"] + 1.4), 3)})
        if n == len(slow) - 1:
            out["shake"].append([round(hit, 3), round(hit + 0.6, 3)])
    return out


def _script_words(line, heard):
    """The line's own words from the script, each timed from what Whisper heard.

    Whisper's spelling is not the script's ("hit" for "hid", digits for numbers),
    and words put on screen must be the script's. So the text comes from the line
    and only the timing from Whisper: each script word takes the time of the best
    matching heard word, or a position proportional to where it sits in the line.
    """
    import difflib
    script = [w for w in re.findall(r"[\w'.-]+", line["text"]) if w.strip(".-")]
    out, j = [], 0
    for k, word in enumerate(script):
        best, best_r = None, 0.0
        for m in range(j, min(len(heard), j + 4)):
            r = difflib.SequenceMatcher(None, word.lower(), heard[m]["word"].lower().strip(".,!?")).ratio()
            if r > best_r:
                best, best_r = m, r
        if best is not None and best_r >= 0.5:
            out.append({"word": word, "start": heard[best]["start"], "end": heard[best]["end"]})
            j = best + 1
        else:
            span = line["end"] - line["start"]
            t = line["start"] + span * k / max(1, len(script))
            out.append({"word": word, "start": round(t, 3), "end": round(t + 0.3, 3)})
    return out


_STOP = set("the a an and or but of to in on at for with that this it is was were be been "
            "you your we our they their who what why how when where there here so then".split())


_WEAK = set("call start look stay find follow said says made make take came went "
             "told tell ever never still just only really very".split())


def _strong_words(line_words, most=2):
    """One or two words worth putting on screen: numbers, names, long words.

    A capital counts as a name only away from the start of the line - "Call" and
    "Start" opening a sentence were being picked as if they mattered.
    """
    def score(k, w):
        word = re.sub(r"[^\w]", "", w["word"])
        if not word or word.lower() in _STOP or word.lower() in _WEAK:
            return -1
        name = word[:1].isupper() and k > 0
        return (3 if word[:1].isdigit() else 0) + (2 if name else 0) + len(word) / 4
    scored = sorted(((score(k, w), k, w) for k, w in enumerate(line_words)),
                    key=lambda x: x[0], reverse=True)
    pick = [w for s, k, w in scored[:most] if s > 1.9]
    return sorted(pick, key=lambda w: w["start"])
