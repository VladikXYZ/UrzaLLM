import json
import os
import time
import unicodedata
from pathlib import Path
import yt_dlp

# ==========================================
# CONFIGURATION
# ==========================================
CHANNEL_URL = "https://www.youtube.com/@KanalSvobodnehoPristavu"
DOWNLOAD_AUDIO = False          # <-- Flip to True when you want to download audio
AUDIO_FORMAT = "mp3"            # "mp3", "m4a", or "wav" (wav/mp3 are great for Whisper + Pyannote)
OUTPUT_JSON = "jsons/channel_metadata.json"
AUDIO_DIR = Path("audio")
ARCHIVE_FILE = "downloaded_audio_archive.txt"
SAVE_EVERY = 10                 # Save JSON progress every N videos

# Keys containing massive arrays of temporary/expiring CDN URLs to exclude from JSON
BLOAT_KEYS = {
    "formats",
    "requested_formats",
    "thumbnails",
    "automatic_captions",
    "subtitles",
    "heatmap",
    "fragments",
}


def clean_Metadata_strings(obj):
    """Recursively normalizes Unicode strings (removes NBSP \u00a0) inside the metadata dict."""
    if isinstance(obj, str):
        return unicodedata.normalize("NFKC", obj).replace("\u00ad", "")
    elif isinstance(obj, list):
        return [clean_Metadata_strings(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: clean_Metadata_strings(v) for k, v in obj.items()}
    return obj


def load_existing_json(json_path: str) -> dict[str, dict]:
    """Loads already scraped metadata keyed by video ID so the script can resume."""
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {item["id"]: item for item in data if "id" in item}
    return {}


def save_json(data_by_id: dict[str, dict], json_path: str):
    """Atomically writes the metadata list to JSON."""
    items = list(data_by_id.values())
    # Sort newest first by timestamp/upload_date
    items.sort(key=lambda x: x.get("timestamp") or 0, reverse=True)
    temp_path = f"{json_path}.tmp"
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=4)
    os.replace(temp_path, json_path)


def get_channel_video_stubs(channel_url: str) -> list[tuple[str, str, str]]:
    """Quickly fetches all video IDs across /videos, /shorts, and /streams."""
    tabs = ["videos", "shorts", "streams"]
    stubs = []
    seen_ids = set()

    flat_opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "ignoreerrors": True,
    }

    with yt_dlp.YoutubeDL(flat_opts) as ydl:
        for tab in tabs:
            tab_url = f"{channel_url.rstrip('/')}/{tab}"
            print(f"Scanning tab: {tab_url} ...")
            info = ydl.extract_info(tab_url, download=False)
            if not info or "entries" not in info:
                continue

            for entry in info["entries"]:
                if not entry or not entry.get("id"):
                    continue
                vid_id = entry["id"]
                if vid_id not in seen_ids:
                    seen_ids.add(vid_id)
                    vid_url = entry.get("url") or f"https://www.youtube.com/watch?v={vid_id}"
                    stubs.append((vid_id, vid_url, tab))

    print(f"Found {len(stubs)} total items across videos, shorts, and streams.")
    return stubs


def scrape_channel(
    channel_url: str = CHANNEL_URL,
    download_audio: bool = DOWNLOAD_AUDIO,
    output_json: str = OUTPUT_JSON,
    audio_dir: Path = AUDIO_DIR,
):
    if download_audio:
        audio_dir.mkdir(parents=True, exist_ok=True)

    scraped_data = load_existing_json(output_json)
    print(f"Loaded {len(scraped_data)} existing items from {output_json}.")

    stubs = get_channel_video_stubs(channel_url)

    # Configure yt-dlp options for per-video full metadata & optional audio
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "ignoreerrors": True,
        "retries": 10,
        "fragment_retries": 10,
        "sleep_interval_requests": 0.5,
        "skip_download": not download_audio,
    }

    if download_audio:
        ydl_opts.update({
            "format": "bestaudio/best",
            "outtmpl": str(audio_dir / "%(id)s.%(ext)s"),
            "download_archive": ARCHIVE_FILE,
            # Skip scheduled/upcoming livestreams that have no audio yet
            "match_filter": yt_dlp.utils.match_filter_func("!is_upcoming & !is_live"),
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": AUDIO_FORMAT,
                    "preferredquality": "192",
                }
            ],
        })

    new_or_updated = 0

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            for idx, (vid_id, vid_url, category_tab) in enumerate(stubs, 1):
                expected_audio_file = audio_dir / f"{vid_id}.{AUDIO_FORMAT}"
                already_has_meta = vid_id in scraped_data
                already_has_audio = expected_audio_file.exists()

                # Skip if we already have everything we need for this video
                if already_has_meta and (not download_audio or already_has_audio):
                    continue

                print(f"[{idx}/{len(stubs)}] Processing {vid_id} ({category_tab})...")
                info = ydl.extract_info(vid_url, download=download_audio)
                if not info:
                    print(f"  -> Skipped {vid_id} (unavailable, private, or upcoming).")
                    continue

                # Convert yt-dlp's internal object into a clean JSON-serializable dict
                clean_info = ydl.sanitize_info(info)
                for key in BLOAT_KEYS:
                    clean_info.pop(key, None)

                # Add custom tracking fields
                clean_info["channel_tab"] = category_tab
                clean_info["audio_file"] = (
                    str(expected_audio_file) if expected_audio_file.exists() else None
                )

                scraped_data[vid_id] = clean_Metadata_strings(clean_info)
                new_or_updated += 1

                if new_or_updated % SAVE_EVERY == 0:
                    save_json(scraped_data, output_json)
                    print(f"  [Checkpoint] Saved {len(scraped_data)} items to {output_json}")

                time.sleep(0.5)

    except KeyboardInterrupt:
        print("\nInterrupted by user! Saving progress before exiting...")
    finally:
        save_json(scraped_data, output_json)
        print(f"Done! {len(scraped_data)} total items saved in {output_json}.")


if __name__ == "__main__":
    scrape_channel()