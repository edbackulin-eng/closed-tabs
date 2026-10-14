"""Upload a finished episode to YouTube: video, thumbnail, subtitles, metadata.

Scheduled rather than published immediately: the video goes up `private` with a
`publishAt`, and YouTube makes it public at that moment. That keeps the channel
on a fixed rhythm regardless of when a build actually finishes, which on free
runners is anywhere in a several-hour window.

The first upload is treated as a test. After its publish moment passes, the
pipeline asks the API whether the video really did turn public, because an
unaudited project holding a video private forever looks exactly like one that
published correctly.
"""
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from core.logger import get_logger

log = get_logger("youtube")

ROOT = Path(__file__).resolve().parent.parent
TOKEN_FILE = ROOT / "youtube_token.json"

CATEGORY_SCIENCE_TECH = "28"
PUBLISH_HOUR_UTC = 15          # evening in Europe, daytime in the US
CHUNK_MB = 8


def _credentials():
    """Build credentials from the environment in CI, or the local file."""
    from google.oauth2.credentials import Credentials

    raw = os.environ.get("YT_TOKEN_JSON")
    data = json.loads(raw) if raw else json.loads(
        TOKEN_FILE.read_text(encoding="utf-8"))
    return Credentials(
        token=None,
        refresh_token=data["refresh_token"],
        client_id=data["client_id"],
        client_secret=data["client_secret"],
        token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
        scopes=data.get("scopes"),
    )


def service():
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=_credentials(), cache_discovery=False)


def channel_name(yt=None):
    yt = yt or service()
    items = yt.channels().list(part="snippet", mine=True).execute().get("items", [])
    return (items[0]["snippet"]["title"], items[0]["id"]) if items else (None, None)


def next_publish_at(last_scheduled=None, now=None, hour=PUBLISH_HOUR_UTC):
    """The slot one day after the last one taken, never in the past.

    Counted from the end of the queue rather than from "now + n days": a runner
    that starts late would otherwise book a slot later than the video before it,
    leaving a hole on the channel.
    """
    now = now or datetime.now(timezone.utc)
    at = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if at <= now + timedelta(minutes=30):
        at += timedelta(days=1)
    if last_scheduled:
        if isinstance(last_scheduled, str):
            last_scheduled = datetime.fromisoformat(
                last_scheduled.replace("Z", "+00:00"))
        after_queue = last_scheduled.replace(
            hour=hour, minute=0, second=0, microsecond=0) + timedelta(days=1)
        at = max(at, after_queue)
    return at


def _parse_metadata(path):
    """Read title, description, tags and hashtags out of metadata.txt."""
    text = Path(path).read_text(encoding="utf-8")
    out, section, buf = {}, None, []

    def flush():
        if section:
            out[section] = "\n".join(buf).strip()

    for line in text.splitlines():
        head = line.strip()
        if head.startswith("TITLE"):
            flush(); section, buf = "title", []
        elif head.startswith("DESCRIPTION"):
            flush(); section, buf = "description", []
        elif head.startswith("TAGS"):
            flush(); section, buf = "tags", []
        elif head.startswith("HASHTAGS"):
            flush(); section, buf = "hashtags", []
        elif head.startswith("UPLOAD SETTINGS"):
            flush(); section, buf = None, []
        elif section:
            buf.append(line)
    flush()
    tags = [t.strip() for t in out.get("tags", "").split(",") if t.strip()]
    return out.get("title", "").strip(), out.get("description", "").strip(), tags


def upload(package_dir, publish_at=None, synthetic_media=True):
    """Upload the package in `package_dir`. Returns the video id."""
    from googleapiclient.http import MediaFileUpload

    package_dir = Path(package_dir)
    video = package_dir / "video.mp4"
    title, description, tags = _parse_metadata(package_dir / "metadata.txt")
    publish_at = publish_at or next_publish_at()

    yt = service()
    name, cid = channel_name(yt)
    log.info("uploading to %s (%s)", name, cid)
    log.info("scheduled for %s", publish_at.strftime("%Y-%m-%d %H:%M UTC"))

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags,
            "categoryId": CATEGORY_SCIENCE_TECH,
            "defaultLanguage": "en",
            "defaultAudioLanguage": "en",
        },
        "status": {
            # private + publishAt is what makes YouTube flip it public on time.
            "privacyStatus": "private",
            "publishAt": publish_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "selfDeclaredMadeForKids": False,
            "containsSyntheticMedia": synthetic_media,
        },
    }
    media = MediaFileUpload(str(video), chunksize=CHUNK_MB * 1024 * 1024,
                            resumable=True, mimetype="video/mp4")
    request = yt.videos().insert(part="snippet,status", body=body, media_body=media)

    response, last = None, -1
    while response is None:
        status, response = request.next_chunk()
        if status and int(status.progress() * 100) >= last + 10:
            last = int(status.progress() * 100)
            log.info("  upload %d%%", last)
    video_id = response["id"]
    log.info("uploaded: https://youtu.be/%s", video_id)

    thumb = package_dir / "thumbnail.png"
    if thumb.exists():
        yt.thumbnails().set(videoId=video_id,
                            media_body=MediaFileUpload(str(thumb))).execute()
        log.info("thumbnail set")

    subs = package_dir / "subs.srt"
    if subs.exists():
        yt.captions().insert(
            part="snippet",
            body={"snippet": {"videoId": video_id, "language": "en",
                              "name": "English", "isDraft": False}},
            media_body=MediaFileUpload(str(subs))).execute()
        log.info("subtitles attached")

    return video_id, publish_at


def is_public(video_id, yt=None):
    """What privacy status YouTube currently reports for a video."""
    yt = yt or service()
    items = yt.videos().list(part="status", id=video_id).execute().get("items", [])
    if not items:
        return None
    return items[0]["status"].get("privacyStatus")


def verify_went_public(video_id, publish_at, grace_minutes=20):
    """After the publish moment, confirm the video actually turned public.

    An unaudited project that silently holds uploads private forever is
    indistinguishable from one that works, until somebody looks. This is the
    look.
    """
    now = datetime.now(timezone.utc)
    due = publish_at + timedelta(minutes=grace_minutes)
    if now < due:
        return {"checked": False,
                "reason": f"not due until {due:%Y-%m-%d %H:%M UTC}"}
    status = is_public(video_id)
    ok = status == "public"
    if not ok:
        log.error("video %s is still '%s' %d minutes after its publish moment. "
                  "If this persists, the project needs the API audit.",
                  video_id, status, grace_minutes)
    return {"checked": True, "status": status, "public": ok}
