import json
from pydoc import text

"""Filtering relevant metada"""
# with open("jsons/channel_metadata.json", "r") as f:
#     yt = json.load(f)
#
# yt_td = [{"title": x["title"], "desc": x["description"], "url": x["webpage_url"]} for x in yt]
# print(yt_td)
# with open("preprocessed/yt.json", "w") as f:
#     json.dump(yt_td, f, indent=1, ensure_ascii=False)

"""Filtering relevant articles"""
with open("jsons/urza_articles.json", "r") as f:
    arts = json.load(f)

with open("preprocessed/yt.json", "r") as f:
    yt = json.load(f)

# print(art)
yt_links = set([x["url"] for x in yt])
print(yt_links)

filtered_arts = []
for art in arts:
    if art["video"] in yt_links:
        if art["text"] != "":
            filtered_arts.append({"title": art["title"], "text": art["text"], "url": art["url"]})
    else:
        if art["text"] != "" and art["title"] != "":
            filtered_arts.append({"title": art["title"], "text": art["text"], "url": art["url"]})

print(len(arts))
print(len(filtered_arts))
with open("preprocessed/arts.json", "w") as f:
    json.dump(filtered_arts, f, indent=1, ensure_ascii=False)