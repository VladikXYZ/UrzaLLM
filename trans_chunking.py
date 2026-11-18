import json
import os
import matplotlib.pyplot as plt
import numpy as np

with open("real_data/yt_full.json") as f:
    yt = json.load(f)

yt_ids = {x["id"]: x for x in yt}
dirr = "real_data/merged_speakers/"
trans = os.listdir(dirr)
chunks = []
total_time = 0
total_words = 0
dist = []
for tr in trans:
    with open(f"{dirr}/{tr}","r") as f:
        data = json.load(f)
    speech = [s for s in data["text"] if s["speaker"] == "Urza"]
    # print(tr)
    # print(len(speech))
    y = yt_ids[f"{tr.split(".")[0]}"]
    for s in speech:
        if len(s["text"].split()) >= 6:
            total_time += s["end"] - s["start"]
            total_words += len(s["text"].split())
            dist.append(len(s["text"].split()))
            chunks.append({"title_cs": y["title_cs"],"subtitle_cs": "", "text_cs": s["text"],
                           "title_en": y["title_en"], "subtitle_en": "", "text_en": s["text"],
                           "src": y["url"], "type": "transcription", "time": s["start"]})
    # print(yt_meta["id"], tr)
print(len(chunks), "chunks")
print(total_time, "seconds")
print(total_words, "words")
print(total_words/len(chunks))
print(np.median(dist))

# plt.figure(figsize=(10, 6))
# plt.hist(dist, bins=50, color='#FF2B2B', edgecolor='black', alpha=0.85)
# plt.title("Distribution of Words per Transcription Chunk", fontsize=14, color='white')
# plt.xlabel("Number of Words", fontsize=12, color='white')
# plt.ylabel("Frequency (Chunk Count)", fontsize=12, color='white')
#
# # Styling for dark theme (matching your Plotly/CachyOS setup)
# plt.gca().set_facecolor('#1e1e2f')
# plt.gcf().patch.set_facecolor('#1e1e2f')
# plt.tick_params(colors='white')
# plt.grid(True, linestyle='--', alpha=0.3)
#
# # Save the figure to disk
# plt.savefig(
#     "real_data/transcription_word_distribution.png",
#     dpi=300,
#     facecolor=plt.gcf().get_facecolor(),
#     edgecolor='none'
# )
# plt.close()  # Clears the figure from memory
print("Histogram saved to real_data/transcription_word_distribution.png")
with open(f"real_data/trans_chunks.json", "w") as f:
    json.dump(chunks, f, indent=1, ensure_ascii=False)