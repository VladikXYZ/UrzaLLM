import json

TAG_MAPPING = {
    # Anarchism variants -> target tag
    "anarchie": "anarchismus",
    "bezstátní společnost": "anarchismus",
    "stateless society": "anarchismus",
    "anarchy": "anarchismus",
    "stát versus anarchie": "anarchismus",

    # Anarchokapitalismus variants
    "ankap": "anarchokapitalismus",
    "anarchokapitalisti": "anarchokapitalismus",
    "anarchokaptialismus": "anarchokapitalismus",
    "anarcho-capitalism": "anarchokapitalismus",
    "anarchokapitalism": "anarchokapitalismus",

    # Anarchoagorismus variants
    "agorismus": "anarchoagorismus",
    "anarchoagorism": "anarchoagorismus",
    "anarcho-agorism": "anarchoagorismus",
    "anarchoaogrismus": "anarchoagorismus",
    "Anarchoagorismmus": "anarchoagorismus",
    "Anarchoagorismus": "anarchoagorismus",

    # NAP variants
    "NAP": "princip neagrese",
    "non-aggression principle": "princip neagrese",
    "neagrese": "princip neagrese",
    "princip neagree": "princip neagrese",

    # Freedom variants
    "freedom": "svoboda",
    "liberty": "svoboda",
    "vnitřní a vnější svoboda": "svoboda",
    "vnitřní svoboda": "svoboda",
    "vnější svoboda": "svoboda",
    "inner freedom": "svoboda",
    "outer liberty": "svoboda",

    # Schooling & Education variants
    "vzdělávání": "školství",
    "školstvo": "školství",
    "education": "školství",
    "svobodné vzdělávání": "sebeřízené vzdělávání",
    "unschooling": "sebeřízené vzdělávání",
    "self directed education": "sebeřízené vzdělávání",

    # Law variants
    "volnotržní právo": "polycentrické právo",
    "decentralizované právo": "polycentrické právo",
    "centrické právo": "polycentrické právo",
    "polycentrické volnotržní právo": "polycentrické právo",

    # Crypto & Money variants
    "bitcoiny": "Bitcoin",
    "BTC": "Bitcoin",
    "kryptopeníze": "kryptoměny",
    "krypto": "kryptoměny",
    "(krypto)peníze": "kryptoměny",
    "XMR": "Monero",

    # Tax variants
    "zdanění": "daně",
}

with open("../jsons/channel_metadata.json", "r") as f:
    raw_meta = json.load(f)

tags = {}
for x in raw_meta:
    for tag in x["tags"]:
        tag = TAG_MAPPING.get(tag, tag)
        tags[tag] = tags.get(tag,0)+1

tags = dict(
    sorted(tags.items(), key=lambda item: item[1], reverse=True)
)

print(tags)
print(len(tags))

# with open("tags.json", "w") as f:
#     json.dump(tags, f, indent=1, ensure_ascii=False)
