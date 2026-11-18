import json
import os

with open("real_data/yt_full.json") as f:
    yt = json.load(f)

yt_ids = {x["id"]: x for x in yt}
dirr = "real_data/merged_speakers/"
trans = os.listdir(dirr)
chunks = []
for tr in trans:
    with open(f"{dirr}/{tr}","r") as f:
        data = json.load(f)
    speech = [s for s in data["text"] if s["speaker"] == "Urza"]
    print(tr)
    print(len(speech))
    y = yt_ids[f"{tr.split(".")[0]}"]
    for s in speech:
        chunks.append({"title_cs": y["title_cs"],"subtitle_cs": "", "text_cs": s["text"],
                       "title_en": y["title_en"], "subtitle_en": "", "text_en": s["text"],
                       "src": y["url"], "type": "transcription", "time": s["start"]})
    # print(yt_meta["id"], tr)
print(len(chunks))
with open(f"real_data/trans_chunks.json", "w") as f:
    json.dump(chunks, f, indent=1, ensure_ascii=False)