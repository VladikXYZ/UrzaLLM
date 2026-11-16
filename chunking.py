import json


def chunk_book(book):
    chunk_cs = []
    chunk_en = []
    for chap in book:
        # print(chap)
        spcs = chap["text_cs"].split("\n\n")
        spen = chap["text_en"].split("\n\n")
        # print(len(spcs) == len(spen))
        if len(spcs) != len(spen):
            print(f"bad count {chap["title_cs"]}: {chap["subtitle_cs"]}")
        for c,e in zip(spcs, spen):
            chunk_cs.append({"title_cs": chap["title_cs"], "subtitle_cs": chap["subtitle_cs"], "text_cs": c})
            chunk_en.append({"title_en": chap["title_en"], "subtitle_en": chap["subtitle_en"], "text_en": e})

with open("materials/anag_full.json") as f:
    anag = json.load(f)

with open("materials/ancap_full.json") as f:
    ancap = json.load(f)

chunk_book(ancap)