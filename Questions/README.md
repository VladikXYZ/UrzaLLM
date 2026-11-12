# Questions: the general political space

`political_space.tsv` is a map of politics independent of Urza, used to measure which political
topics his texts address (report: `Projection/political_space.py`).

| Column | Meaning |
|---|---|
| `id` | `T..` subtopic, `Q..` question, `C..` concept |
| `layer` | `topic`, `question` or `concept` |
| `major` | policy area (Comparative Agendas Project topics, adapted to Czech politics, plus “Morálka a osobní svoboda”); for concepts their group |
| `sub` | subtopic the item belongs to |
| `text` | the topic name, question or concept |
| `description` | short description of a topic or concept (embedded together with the name) |

Edit the file freely (add questions, rephrase, add areas), then rerun from the repo root:

```bash
python Questions/embed_questions.py
python Projection/political_space.py
```

Embedding 326 short texts takes about 20 seconds on a CPU (needs `torch` and
`sentence-transformers`, see `Embedding/requirements.txt`).
