"""YouTube for Closed Tabs: upload, schedule, captions, thumbnail, branding.

Every upload goes up private with a publishAt, on the nearest free slot of the
channel's week (Mon, Tue, Thu, Fri, Sat at 17:00 UTC) that is at least two
hours away. Every video declares containsSyntheticMedia=true and madeForKids=false
and sits in Entertainment.

A video is never uploaded twice. The moment YouTube returns an id it is written
to uploaded.json beside the episode, and the upload stage refuses to run for any
episode that already has one, whatever the episode database says - the stoicism
channel uploaded the same video twice when its database push was lost.

Scopes: youtube.upload to upload, youtube (full) to set captions, change privacy
and edit branding.
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
SCOPES = ["https://www.googleapis.com/auth/youtube.upload",
          "https://www.googleapis.com/auth/youtube"]

CATEGORY_ENTERTAINMENT = "24"
SLOT_WEEKDAYS = (0, 1, 3, 4, 5)        # Mon, Tue, Thu, Fri, Sat
SLOT_HOUR_UTC = 17
MIN_LEAD = timedelta(hours=2)
CHUNK_MB = 8


# --- auth ----------------------------------------------------------------------

def _credentials():
    from google.oauth2.credentials import Credentials
    raw = os.environ.get("YT_TOKEN_JSON")
    data = json.loads(raw) if raw else json.loads(TOKEN_FILE.read_text(encoding="utf-8"))
    return Credentials(token=None, refresh_token=data["refresh_token"],
                       client_id=data["client_id"], client_secret=data["client_secret"],
                       token_uri=data.get("token_uri", "https://oauth2.googleapis.com/token"),
                       scopes=data.get("scopes") or SCOPES)


def service():
    from googleapiclient.discovery import build
    return build("youtube", "v3", credentials=_credentials(), cache_discovery=False)


def authorize(client_secret, token_out=TOKEN_FILE):
    """Local browser consent; writes the refresh token. Run once, by hand."""
    from google_auth_oauthlib.flow import InstalledAppFlow
    flow = InstalledAppFlow.from_client_secrets_file(str(client_secret), SCOPES)
    creds = flow.run_local_server(port=0, prompt="consent", access_type="offline")
    data = {"refresh_token": creds.refresh_token, "client_id": creds.client_id,
            "client_secret": creds.client_secret, "token_uri": creds.token_uri,
            "scopes": SCOPES}
    Path(token_out).write_text(json.dumps(data), encoding="utf-8")
    return data


def channel(yt=None):
    yt = yt or service()
    items = yt.channels().list(part="snippet,contentDetails,brandingSettings,statistics",
                               mine=True).execute().get("items", [])
    return items[0] if items else None


# --- the schedule ----------------------------------------------------------------

def next_slot(taken=(), now=None):
    """The nearest free publishing slot at least two hours from now.

    `taken` is every publishAt already handed to YouTube. A slot is free when no
    video of ours already holds that day.
    """
    now = now or datetime.now(timezone.utc)
    taken_days = set()
    for t in taken:
        if isinstance(t, str):
            t = datetime.fromisoformat(t.replace("Z", "+00:00"))
        taken_days.add(t.astimezone(timezone.utc).date())
    day = now.replace(hour=SLOT_HOUR_UTC, minute=0, second=0, microsecond=0)
    for _ in range(60):
        if (day.weekday() in SLOT_WEEKDAYS and day - now >= MIN_LEAD
                and day.date() not in taken_days):
            return day
        day += timedelta(days=1)
    raise RuntimeError("no free slot in the next 60 days")


# --- upload ----------------------------------------------------------------------

def body(title, description, tags, publish_at, language="en"):
    return {
        "snippet": {"title": title[:100], "description": description[:5000],
                    "tags": tags, "categoryId": CATEGORY_ENTERTAINMENT,
                    "defaultLanguage": language, "defaultAudioLanguage": language},
        "status": {"privacyStatus": "private",
                   "publishAt": publish_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "selfDeclaredMadeForKids": False,
                   "containsSyntheticMedia": True},
    }


def upload(episode_dir, title, description, tags, publish_at, yt=None):
    """Upload video.mp4 from episode_dir. Returns the video id.

    Refuses when uploaded.json already exists, and writes it before anything else
    can fail."""
    from googleapiclient.http import MediaFileUpload
    episode_dir = Path(episode_dir)
    marker = episode_dir / "uploaded.json"
    if marker.exists():
        saved = json.loads(marker.read_text(encoding="utf-8"))
        log.warning("already uploaded as %s; not uploading again", saved["youtube_id"])
        return saved["youtube_id"]
    yt = yt or service()
    media = MediaFileUpload(str(episode_dir / "video.mp4"), chunksize=CHUNK_MB * 1024 * 1024,
                            resumable=True, mimetype="video/mp4")
    req = yt.videos().insert(part="snippet,status",
                             body=body(title, description, tags, publish_at), media_body=media)
    response, last = None, -1
    while response is None:
        status, response = req.next_chunk()
        if status and int(status.progress() * 100) >= last + 10:
            last = int(status.progress() * 100)
            log.info("  upload %d%%", last)
    vid = response["id"]
    marker.write_text(json.dumps({"youtube_id": vid,
                                  "publish_at": publish_at.strftime("%Y-%m-%dT%H:%M:%SZ")}),
                      encoding="utf-8")
    log.info("uploaded https://youtu.be/%s for %s", vid, publish_at.isoformat())
    return vid


def attach(video_id, episode_dir, yt=None):
    """Thumbnail and English subtitles."""
    from googleapiclient.http import MediaFileUpload
    yt = yt or service()
    episode_dir = Path(episode_dir)
    thumb = episode_dir / "thumbnail.jpg"
    if thumb.exists():
        yt.thumbnails().set(videoId=video_id, media_body=MediaFileUpload(str(thumb))).execute()
        log.info("thumbnail set")
    srt = episode_dir / "subs.srt"
    if srt.exists():
        yt.captions().insert(part="snippet",
                             body={"snippet": {"videoId": video_id, "language": "en",
                                               "name": "English", "isDraft": False}},
                             media_body=MediaFileUpload(str(srt), mimetype="application/octet-stream"),
                             ).execute()
        log.info("subtitles attached")


def status_of(video_id, yt=None):
    yt = yt or service()
    items = yt.videos().list(part="status", id=video_id).execute().get("items", [])
    return items[0]["status"] if items else None


# --- the channel ---------------------------------------------------------------------

def all_uploads(yt=None):
    """Every video on the channel: (id, title, privacy)."""
    yt = yt or service()
    ch = channel(yt)
    playlist = ch["contentDetails"]["relatedPlaylists"]["uploads"]
    out, token = [], None
    while True:
        r = yt.playlistItems().list(part="contentDetails,snippet", playlistId=playlist,
                                    maxResults=50, pageToken=token).execute()
        ids = [i["contentDetails"]["videoId"] for i in r.get("items", [])]
        if ids:
            for v in yt.videos().list(part="status,snippet", id=",".join(ids)).execute()["items"]:
                out.append((v["id"], v["snippet"]["title"], v["status"]["privacyStatus"]))
        token = r.get("nextPageToken")
        if not token:
            return out


def make_all_private(yt=None, dry_run=False):
    """Every existing video to private. Nothing is deleted."""
    yt = yt or service()
    changed = []
    for vid, title, privacy in all_uploads(yt):
        if privacy == "private":
            continue
        if not dry_run:
            v = yt.videos().list(part="status", id=vid).execute()["items"][0]
            st = v["status"]
            st["privacyStatus"] = "private"
            st.pop("publishAt", None)
            yt.videos().update(part="status", body={"id": vid, "status": st}).execute()
            time.sleep(0.5)
        changed.append((vid, title, privacy))
    return changed


def set_branding(description, keywords, banner_path=None, yt=None):
    """Channel description, keywords and banner. The profile picture cannot be set
    through the Data API; it is uploaded in YouTube Studio."""
    from googleapiclient.http import MediaFileUpload
    yt = yt or service()
    ch = channel(yt)
    settings = ch.get("brandingSettings", {})
    settings.setdefault("channel", {})
    settings["channel"]["description"] = description
    settings["channel"]["keywords"] = keywords
    if banner_path:
        r = yt.channelBanners().insert(media_body=MediaFileUpload(str(banner_path))).execute()
        settings.setdefault("image", {})["bannerExternalUrl"] = r["url"]
    yt.channels().update(part="brandingSettings",
                         body={"id": ch["id"], "brandingSettings": settings}).execute()
    return ch["id"]
