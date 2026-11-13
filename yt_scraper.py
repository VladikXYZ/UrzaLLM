from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import threading
import yt_dlp

# --- CONFIGURATION ---
CHANNEL_URL = "https://www.youtube.com/@KanalSvobodnehoPristavu"
DOWNLOAD_AUDIO = True  # <-- Toggle True/False
OUTPUT_JSON = Path("mat_czech/yt_metada.json")
AUDIO_DIR = Path("audio")
ARCHIVE_FILE = "downloaded_audio_archive.txt"
SAVE_EVERY = 10
MAX_WORKERS = 1  # Number of concurrent threads

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


# --- FETCH VIDEO STUBS ---
stubs = []
seen = set()

print("Fetching video list from channel tabs...")
with yt_dlp.YoutubeDL(
    {"extract_flat": "in_playlist", "quiet": True, "ignoreerrors": True}
) as ydl:
  for tab in ["videos", "shorts", "streams"]:
    print(f" -> Looking at {tab}")
    info = ydl.extract_info(f"{CHANNEL_URL.rstrip('/')}/{tab}", download=False)
    for entry in (info or {}).get("entries") or []:
      if entry and entry.get("id") and entry["id"] not in seen:
        vid_id = entry["id"]
        seen.add(vid_id)
        stubs.append((vid_id, f"https://www.youtube.com/watch?v={vid_id}", tab))

print(f"Found {len(stubs)} total unique videos.")

scraped_data = load_checkpoint(OUTPUT_JSON)
print(f"Loaded {len(scraped_data)} existing items from checkpoint.")

# --- PRE-FILTER TASKS (Skip already processed) ---
tasks = []
for vid_id, vid_url, tab in stubs:
  has_meta = vid_id in scraped_data
  has_audio = (
      any(AUDIO_DIR.glob(f"{vid_id}.*")) if DOWNLOAD_AUDIO else True
  )
  if not (has_meta and has_audio):
    tasks.append((vid_id, vid_url, tab))

print(f"Remaining tasks to process: {len(tasks)}")

# --- WORKER FUNCTION ---
lock = threading.Lock()
new_count = 0


def process_video(vid_id, vid_url, tab):
  ydl_opts = {
      "quiet": True,
      "no_warnings": True,
      "ignoreerrors": True,
      "skip_download": not DOWNLOAD_AUDIO,
  }

  if DOWNLOAD_AUDIO:
    ydl_opts.update({
        "format": "bestaudio/best",
        "outtmpl": str(AUDIO_DIR / "%(id)s.%(ext)s"),
        "download_archive": ARCHIVE_FILE,
        "match_filter": yt_dlp.utils.match_filter_func(
            "!is_upcoming & !is_live"
        ),
    })

  try:
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      info = ydl.extract_info(vid_url, download=DOWNLOAD_AUDIO)
      if not info:
        return None
      return vid_id, ydl.sanitize_info(info)
  except Exception as e:
    print(f"Error processing {vid_id}: {e}")
    return None


# --- MULTITHREADED EXECUTION ---
try:
  with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    future_to_video = {
        executor.submit(process_video, vid_id, vid_url, tab): (
            vid_id,
            vid_url,
            tab,
        )
        for vid_id, vid_url, tab in tasks
    }

    for i, future in enumerate(as_completed(future_to_video), 1):
      vid_id, _, tab = future_to_video[future]
      try:
        result = future.result()
        if result:
          v_id, sanitized_info = result
          with lock:
            scraped_data[v_id] = sanitized_info
            new_count += 1
            print(f"[{i}/{len(tasks)}] Processed {v_id} ({tab})")

            if new_count % SAVE_EVERY == 0:
              save_checkpoint(scraped_data, OUTPUT_JSON)
              print(f"  -> Checkpoint saved ({len(scraped_data)} items)")
      except Exception as exc:
        print(f"Video {vid_id} generated an exception: {exc}")

except KeyboardInterrupt:
  print("\nInterrupted by user! Saving progress...")
finally:
  save_checkpoint(scraped_data, OUTPUT_JSON)
  print(f"Finished. Total saved in checkpoint: {len(scraped_data)}")