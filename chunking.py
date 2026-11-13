import json

with open("english/anag_tr.json") as f:
    anag = json.load(f)
with open("english/ancap_tr.json") as f:
    ancap = json.load(f)


with open("english/anag_en.json") as f:
    anagen = json.load(f)
with open("english/ancap_en.json") as f:
    ancapen = json.load(f)

chunks = []
en_chunks = []

en_concepts = []
concepts = []
for chapter in anag:
    splitted = chapter["text"].split("\n\n")
    # print(splitted)
    concept = chapter["name"] if chapter["chapter"] == "" else chapter["chapter"]
    concepts.append({"type": "concept", "title": concept, "subtitle": "", "text": concept, "source": "AG"})
    for chunk in splitted:
        if chapter["chapter"] == "":
            title = chapter["name"]
            subtitle = ""
        else:
            title = chapter["chapter"]
            subtitle = chapter["name"]
        chunks.append({"type": "book", "title": title, "subtitle": subtitle, "text": chunk, "source": "AG"})

for chapter in ancap:
    splitted = chapter["text"].split("\n\n")
    # print(splitted)
    concept = chapter["name"] if chapter["chapter"] == "" else chapter["chapter"]
    concepts.append({"type": "concept", "title": concept, "subtitle": "", "text": concept, "source": "AC"})
    for chunk in splitted:
        if chapter["chapter"] == "":
            title = chapter["name"]
            subtitle = ""
        else:
            title = chapter["chapter"]
            subtitle = chapter["name"]
        chunks.append({"type": "book", "title": title, "subtitle": subtitle, "text": chunk, "source": "AC"})

for chapter in anagen:
    splitted = chapter["text"].split("\n\n")
    # print(splitted)
    concept = chapter["name"] if chapter["chapter"] == "" else chapter["chapter"]
    en_concepts.append({"type": "concept", "title": concept, "subtitle": "", "text": concept, "source": "AG"})
    for chunk in splitted:
        if chapter["chapter"] == "":
            title = chapter["name"]
            subtitle = ""
        else:
            title = chapter["chapter"]
            subtitle = chapter["name"]
        en_chunks.append({"type": "book", "title": title, "subtitle": subtitle, "text": chunk, "source": "AG"})

for chapter in ancapen:
    splitted = chapter["text"].split("\n\n")
    # print(splitted)
    concept = chapter["name"] if chapter["chapter"] == "" else chapter["chapter"]
    en_concepts.append({"type": "concept", "title": concept, "subtitle": "", "text": concept, "source": "AC"})
    for chunk in splitted:
        if chapter["chapter"] == "":
            title = chapter["name"]
            subtitle = ""
        else:
            title = chapter["chapter"]
            subtitle = chapter["name"]
        en_chunks.append({"type": "book", "title": title, "subtitle": subtitle, "text": chunk, "source": "AC"})

with open("english/arts.json") as f:
    arts = json.load(f)

for art in arts:
    chunks.append({"type": "article", "title": art["title"], "subtitle": "", "text": art["text"], "source": art["url"]})

# with open("english/arts_en.json") as f:
#     trarts = json.load(f)
#
# enarts = []
# for tr, a in zip(trarts, arts):
#     tempart = a
#     tempart["title"] = tr[0]
#     tempart["text"] = tr[1]
#     enarts.append(tempart)
#
# with open("english/arts_en_full.json", "w") as f:
#     json.dump(enarts, f, indent=1, ensure_ascii=False)

with open("english/arts_en_full.json") as f:
    enarts = json.load(f)
for art in enarts:
    en_chunks.append({"type": "article", "title": art["title"], "subtitle": "", "text": art["text"], "source": art["url"]})


with open("english/yt.json") as f:
    yt = json.load(f)
with open("english/yt_en_full.json") as f:
    enyt = json.load(f)

# enyts_corr = []
# for c,e in zip(yt,enyt):
#     en = c
#     en["title"] = e[0]
#     en["desc"] = e[1]
#     enyts_corr.append(en)
#
# with open("english/yt_en_full.json", "w") as f:
#     json.dump(enyts_corr, f, indent=1, ensure_ascii=False)


for vid in yt:
    chunks.append({"type": "video", "title": vid["title"], "subtitle": "", "text": vid["desc"], "source": vid["url"]})

for vid in enyt:
    en_chunks.append({"type": "video", "title": vid["title"], "subtitle": "", "text": vid["desc"], "source": vid["url"]})

chunks.extend(concepts)
en_chunks.extend(en_concepts)
print(len(chunks))
print(len(en_chunks))
with open("chunked/chunks.json","w") as f:
    json.dump(chunks, f, indent=1, ensure_ascii=False)
with open("chunked/en_chunks.json","w") as f:
    json.dump(chunks, f, indent=1, ensure_ascii=False)