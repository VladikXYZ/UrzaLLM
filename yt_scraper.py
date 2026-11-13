import json
from pathlib import Path
import yt_dlp

# --- CONFIGURATION ---
CHANNEL_URL = "https://www.youtube.com/@KanalSvobodnehoPristavu"
DOWNLOAD_AUDIO = False  # <-- Toggle True/False
OUTPUT_JSON = Path("jsons/yt_data.json")
AUDIO_DIR = Path("audio")
ARCHIVE_FILE = "downloaded_audio_archive.txt"
SAVE_EVERY = 10

OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
if DOWNLOAD_AUDIO:
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# --- CHECKPOINT HELPERS ---
def load_checkpoint(path: Path) -> dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return {item["id"]: item for item in json.load(f) if "id" in item}
    return {}

def save_checkpoint(data: dict, path: Path):
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(list(data.values()), f, ensure_ascii=False, indent=1)
    tmp.replace(path)



stubs = []
seen = set()

with yt_dlp.YoutubeDL({"extract_flat": "in_playlist", "quiet": True, "ignoreerrors": True}) as ydl:
    for tab in ["videos", "shorts", "streams"]:
        print(f"Looking at {tab}")
        info = ydl.extract_info(f"{CHANNEL_URL.rstrip('/')}/{tab}", download=False)
        for entry in (info or {}).get("entries") or []:
            if entry and entry.get("id") and entry["id"] not in seen:
                vid_id = entry["id"]
                seen.add(vid_id)
                stubs.append((vid_id, f"https://www.youtube.com/watch?v={vid_id}", tab))

print(f"Found {len(stubs)} total videos.")

scraped_data = load_checkpoint(OUTPUT_JSON)
print(f"Loaded {len(scraped_data)} existing items from checkpoint.")

ydl_opts = {
    "quiet": True,
    "no_warnings": True,
    "ignoreerrors": True,
    "skip_download": not DOWNLOAD_AUDIO,
}

if DOWNLOAD_AUDIO:
    ydl_opts.update({
        "format": "bestaudio/best",  # Downloads raw audio (.webm/.m4a) with no FFmpeg postprocessing
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "download_archive": ARCHIVE_FILE,
        "match_filter": yt_dlp.utils.match_filter_func("!is_upcoming & !is_live"),
    })

new_count = 0

try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        for i, (vid_id, vid_url, tab) in enumerate(stubs, 1):
            has_meta = vid_id in scraped_data
            has_audio = any(AUDIO_DIR.glob(f"{vid_id}.*")) if DOWNLOAD_AUDIO else True

            if has_meta and has_audio:
                continue

            print(f"[{i}/{len(stubs)}] Processing {vid_id} ({tab})...")
            info = ydl.extract_info(vid_url, download=DOWNLOAD_AUDIO)
            if not info:
                continue

            # sanitize_info only makes yt-dlp internal objects JSON-serializable
            scraped_data[vid_id] = ydl.sanitize_info(info)
            new_count += 1

            if new_count % SAVE_EVERY == 0:
                save_checkpoint(scraped_data, OUTPUT_JSON)
                print(f"  -> Checkpoint saved ({len(scraped_data)} items)")

except KeyboardInterrupt:
    print("Interrupted! Saving progress...")
finally:
    save_checkpoint(scraped_data, OUTPUT_JSON)
    print(f"Finished. Total saved: {len(scraped_data)}")