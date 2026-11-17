import json

"""BOOKS"""
def merge_book(name):
    with open(f"mat_czech/{name}.json") as f:
        anagcs = json.load(f)

    with open(f"better_translation/{name}_en.json") as f:
        anagen = json.load(f)

    merged = []
    for c,e in zip(anagcs, anagen):
        # print(c)
        # print(e)
        chapter_cs = c["name"] if c["chapter"] == "" else c["chapter"]
        chapter_cs = chapter_cs.split(" ›")[0]
        chapter_en = e["name"] if e["chapter"] == "" else e["chapter"]
        # print(chapter_cs, chapter_en)
        sub_cs = "" if c["chapter"] == "" else c["name"]
        sub_en = "" if e["chapter"] == "" else e["name"]
        # print(sub_cs, sub_en)
        chap = {"title_cs": chapter_cs, "title_en": chapter_en, "subtitle_cs": sub_cs,
                "subtitle_en": sub_en, "text_cs": c["text"], "text_en": e["text"]}
        merged.append(chap)

    with open(f"materials/{name}_full.json", "w") as f:
        json.dump(merged, f, indent=1, ensure_ascii=False)

merge_book("ancap")
merge_book("anag")

"""ARTICLES"""
with open("mat_czech/arts_filtered.json") as f:
    artcs = json.load(f)

with open("better_translation/arts_en.json") as f:
    arten = json.load(f)
merged = []
print(len(artcs) == len(arten))
for c,e in zip(artcs, arten):
    c["title_cs"] = c.pop("title")
    c["text_cs"] = c.pop("text")
    c["title_en"] = e[0]
    c["text_en"] = e[1]
    merged.append(c)


with open("materials/arts_full.json", "w") as f:
    json.dump(merged, f, indent=1, ensure_ascii=False)

"""YT"""
# with open("mat_czech/yt_metadata_nice.json") as f:
#     ytcs = json.load(f)
#
# with open("mat_translated/yt_en.json") as f:
#     yten = json.load(f)
# merged = []
# print(len(ytcs))
# print(len(yten))
# print(len(ytcs) == len(yten))
# for c,e in zip(ytcs, yten):
#     c["title_cs"] = c.pop("title")
#     c["desc_cs"] = c.pop("description")
#     c["title_en"] = e[0]
#     c["desc_en"] = e[1]
#     c.pop("transcript")
#     merged.append(c)
#
#
# with open("materials/yt_full.json", "w") as f:
#     json.dump(merged, f, indent=1, ensure_ascii=False)

