import json
from pydoc import text

"""Filtering relevant metada"""
# with open("jsons/yt_data.json", "r") as f:
#     yt = json.load(f)
#
# yt_td = [{"title": x["title"], "desc": x["description"], "url": x["webpage_url"]} for x in yt]
# print(yt_td)
# with open("preprocessed/yt.json", "w") as f:
#     json.dump(yt_td, f, indent=1, ensure_ascii=False)

"""Filtering relevant articles"""
# with open("jsons/arts.json", "r") as f:
#     arts = json.load(f)
#
# with open("preprocessed/yt.json", "r") as f:
#     yt = json.load(f)
#
# # print(art)
# yt_links = set([x["url"] for x in yt])
# print(yt_links)
#
# filtered_arts = []
# for art in arts:
#     if art["video"] in yt_links:
#         if art["text"] != "":
#             # filtered_arts.append({"title": art["title"], "text": art["text"], "url": art["url"]})
#             filtered_arts.append(art)
#     else:
#         if art["text"] != "" and art["title"] != "":
#             # filtered_arts.append({"title": art["title"], "text": art["text"], "url": art["url"]})
#             filtered_arts.append(art)
#
#
# print(len(arts))
# print(len(filtered_arts))
# with open("preprocessed/arts.json", "w") as f:
#     json.dump(filtered_arts, f, indent=1, ensure_ascii=False)


"""Preparing for translation"""
# with open("preprocessed/yt.json", "r") as f:
#     yt = json.load(f)
# with open("preprocessed/arts.json", "r") as f:
#     arts = json.load(f)
#
# yt_tr = [(x["title"], x["desc"]) for x in yt]
# arts_tr = [(x["title"], x["text"]) for x in arts]
#
# with open("preprocessed/arts_tr.json", "w") as f:
#     json.dump(arts_tr, f, indent=1, ensure_ascii=False)
# with open("preprocessed/yt_tr.json", "w") as f:
#     json.dump(yt_tr, f, indent=1, ensure_ascii=False)


"""Chapter filtering"""
with open("jsons/ankap.json", "r") as f:
    ancap = json.load(f)

print(ancap[34])
for x in ancap[34:]:
    splitted = x["text"].split("\n\n")
    x["name"] = splitted[0]
    x["text"] = "\n\n".join(splitted[1:])

with open("preprocessed/ancap_tr.json", "w") as f:
    json.dump(ancap, f, indent=1, ensure_ascii=False)



