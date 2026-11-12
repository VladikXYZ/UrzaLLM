import json

with open("preprocessed/anag_tr.json") as f:
    anag = json.load(f)
with open("preprocessed/ancap_tr.json") as f:
    ancap = json.load(f)


with open("preprocessed/anag_en.json") as f:
    anagen = json.load(f)
with open("preprocessed/ancap_en.json") as f:
    ancapen = json.load(f)

chunks = []
en_chunks = []
for chapter in anag:
    splitted = chapter["text"].split("\n\n")
    # print(splitted)
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
    for chunk in splitted:
        if chapter["chapter"] == "":
            title = chapter["name"]
            subtitle = ""
        else:
            title = chapter["chapter"]
            subtitle = chapter["name"]
        en_chunks.append({"type": "book", "title": title, "subtitle": subtitle, "text": chunk, "source": "AC"})

with open("preprocessed/arts.json.json") as f:
    arts = json.load(f)

with open("preprocessed/chunks.json","w") as f:
    json.dump(chunks, f, indent=1, ensure_ascii=False)