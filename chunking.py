import json


def chunk_book(name):
    chunked = []
    with open(f"materials/{name}_full.json") as f:
        book = json.load(f)
    for chap in book:
        # print(chap)
        spcs = chap["text_cs"].split("\n\n")
        spen = chap["text_en"].split("\n\n")
        # print(len(spcs) == len(spen))
        if len(spcs) != len(spen):
            print(len(spcs))
            print(len(spen))
            print(f"bad count {chap["title_cs"]}: {chap["subtitle_cs"]}")
        for c,e in zip(spcs, spen):
            chunked.append({"title_cs": chap["title_cs"], "subtitle_cs": chap["subtitle_cs"], "text_cs": c, "title_en": chap["title_en"], "subtitle_en": chap["subtitle_en"], "text_en": e, "src": "", "type": name})
    return chunked



with open("chunked/chunks.json") as f:
    cs_chunks = json.load(f)

cs_chunks = [x for x in cs_chunks if x["type"] != "concept"]

old_books = [x for x in cs_chunks if x["type"] == "book"]
old_arts = [x for x in cs_chunks if x["type"] == "article"]
old_vids = [x for x in cs_chunks if x["type"] == "video"]

chunks = chunk_book("ancap")
chunks.extend(chunk_book("anag"))

# print(f"Books: {len(all_cs)} vs {len(old_books)}")

with open("materials/yt_full.json") as f:
    yt = json.load(f)

with open("materials/arts_full.json") as f:
    arts = json.load(f)


for y in yt:
    chunks.append({"title_cs": y["title_cs"],"subtitle_cs": "", "text_cs": y["desc_cs"], "title_en": y["title_en"], "subtitle_en": "", "text_en": y["desc_en"], "src": y["url"], "type": "article"})

for a in arts:
    chunks.append({"title_cs": a["title_cs"],"subtitle_cs": "", "text_cs": a["text_cs"], "title_en": a["title_en"], "subtitle_en": "", "text_en": a["text_en"], "src": a["url"], "type": "video"})


with open("chunked/chunked.json", "w") as f:
    json.dump(chunks, f, indent=1, ensure_ascii=False)



print(len(chunks) == len(cs_chunks))
print(len(chunks), len(cs_chunks))