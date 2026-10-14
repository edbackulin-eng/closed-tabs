"""Authorise the channel once, by hand, and store the token as a GitHub secret.

    python scripts/youtube_auth.py client_secret.json            # consent, show channel
    python scripts/youtube_auth.py --store                       # put the token in Secrets
    python scripts/youtube_auth.py --list                        # every video and its privacy
    python scripts/youtube_auth.py --make-private                # all existing videos to private

The consent screen opens in the browser. Pick the channel being converted to
Closed Tabs, then check the channel name and id this prints before --store.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from pipeline import youtube  # noqa: E402

REPO = "edbackulin-eng/closed-tabs"


def main(argv):
    if "--store" in argv:
        token = youtube.TOKEN_FILE.read_text(encoding="utf-8")
        subprocess.run(["gh", "secret", "set", "YT_TOKEN_JSON", "--repo", REPO],
                       input=token, text=True, check=True)
        print("YT_TOKEN_JSON stored in", REPO)
        return 0
    if "--list" in argv:
        for vid, title, privacy in youtube.all_uploads():
            print(f"{privacy:9s} {vid}  {title}")
        return 0
    if "--make-private" in argv:
        changed = youtube.make_all_private()
        for vid, title, was in changed:
            print(f"{was:9s} -> private  {vid}  {title}")
        print(f"{len(changed)} video(s) made private; nothing deleted")
        return 0
    youtube.authorize(Path(argv[0]))
    ch = youtube.channel()
    print(f"channel: {ch['snippet']['title']}")
    print(f"id:      {ch['id']}")
    print(f"handle:  {ch['snippet'].get('customUrl', '-')}")
    print(f"videos:  {ch['statistics'].get('videoCount')}")
    print("If this is the right channel: python scripts/youtube_auth.py --store")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
