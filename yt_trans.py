import json
from pathlib import Path

src_dir = Path("better_translation/urza_translation_package")
dst_dir = Path("better_translation/transcriptions")

dst_dir.mkdir(parents=True, exist_ok=True)

processed_files = []

# Iterate over all subfolders, copy and transform sentences.json -> <parent_directory>.json
for folder in sorted(src_dir.iterdir()):
    if folder.is_dir():
        sentences_file = folder / "sentences.json"
        if sentences_file.exists():
            with open(sentences_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Update sentence keys: 'text' -> 'text_cs', and replace 'translate' with 'text_en': ""
            for s in data.get("sentences", []):
                text_val = s.pop("text", "")
                if "translate" in s:
                    del s["translate"]
                s["text_cs"] = text_val
                s["text_en"] = ""

            target_file = dst_dir / f"{folder.name}.json"
            with open(target_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, ensure_ascii=False)

            processed_files.append(target_file.name)

print(f"Successfully processed {len(processed_files)} files in {dst_dir}")
