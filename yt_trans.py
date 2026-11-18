import json
import os

dirr = "real_data/urza_translation_package2/"
trans = os.listdir(dirr)
print(trans)
for tr in trans:
    if tr in ["index.json", "README.md"]:
        continue
    with open(f"{dirr}/{tr}","r") as f:
        data = json.load(f)
    # speech = [s for s in data["sentences"] if s["speaker"] == "Urza"]
    speech = data["sentences"]
    print(tr)
    print(len(speech))
    urza = {"id":data["video_id"], "src": data["url"], "title": data["title"]}
    if len(speech)>= 1:
        merged = [speech[0]]
        for s in speech[1:]:
            last = merged[-1]
            if last["speaker"] == s["speaker"] and len(last["text"]) + len(s["text"]) <= 400:
                last["text"] += " "+s["text"]
                last["end"] = s["end"]
            else:
                merged.append(s)
        print(len(merged))
        urza["text"] = merged
        with open(f"real_data/merged_speakers/{tr}", "w") as f:
            json.dump(urza, f, indent=1, ensure_ascii=False)