# Urza video transcripts — translation into English

968 videos from the YouTube channel *Kanál Svobodného přístavu* (talks, interviews and debates with the
Czech anarcho-capitalist Urza), transcribed automatically and split into numbered **sentences**.
About **890,000 sentences / 14.6 million words** of Czech to translate.

```
README.md          this file
index.json         every video: title, date, number of sentences and words to translate
<video_id>.json    968 files, one per video: the sentences to translate (the input)
```

Video `<video_id>` is `https://www.youtube.com/watch?v=<video_id>`. YouTube ids can start with `-`
(7 files here, e.g. `-jANazuSiRQ.json`): in shell commands write `./-jANazuSiRQ.json` or put `--` before it.

## Input: `<video_id>.json`

```json
{
  "video_id": "Q0tFQlGWdlY", "title": "Urza versus asi 15 etatistů", "date": "2025-06-28",
  "url": "https://www.youtube.com/watch?v=Q0tFQlGWdlY", "duration": 12933.38,
  "n_sentences": 2855, "n_to_translate": 2855, "words_to_translate": 36390,
  "sentences": [
    {"id": "s00044", "start": 128.49, "end": 131.33, "speaker": "Urza", "language": "cs",
     "translate": true, "text": "Já jsem nevěděl, že budeš říct tohle, stát je nemorální."},
    {"id": "s00045", "start": 133.03, "end": 133.93, "speaker": "Urza", "language": "cs",
     "translate": true, "text": "Já jsem to fakt..."}
  ]
}
```

- **`id`**: the sentence's number within the video. The translation is matched back by it, so keep it
  exactly.
- **`speaker`**: `"Urza"`, or another person (`speaker_0`, `speaker_1`, … numbered per video). Useful for
  dialogue: who is talking to whom.
- **`translate`**: `false` for sentences already spoken in English. Leave those out (or copy them as they are).
- `start`/`end`: seconds in the video, if you want to listen.

## Output: a folder `translations/` with one file per video, same name as the input (`<video_id>.json`)

```json
{"video_id": "Q0tFQlGWdlY",
 "model": "which model/agents translated it",
 "sentences": [
   {"id": "s00044", "en": "I didn't know you'd say this, that the state is immoral."},
   {"id": "s00045", "en": "I really..."}
 ]}
```

**Exactly one English sentence per input sentence, same `id`**, so we can put any span of sentences
back together. Please don't merge or split sentences, even when one is just a fragment ("A proč..." →
"And why..."); the fragment is half of a thought that continues in the next sentence.

## How to translate well

- **Translate in context, not sentence by sentence.** Give the model a run of consecutive sentences
  (e.g. 50–150, a few minutes of speech) and ask for the translation of each `id`. Sentences are short
  (median 10 words) and often continue each other; alone, many are ambiguous.
- **Spoken, colloquial Czech** (`bejt`, `těma`, `jsou tam takový lidi`, `jakože`): translate into natural
  spoken English, keep the tone; don't smooth it into an essay. Fillers ("no", "jako", "prostě") can be
  dropped or kept light.
- **Recognition errors exist** (automatic transcription, ~2–5% of words; names suffer most). If a word
  makes no sense, translate the most plausible meaning; don't invent content.
- **Names and terms** that come up a lot:

  | Czech | English |
  |---|---|
  | Urza | Urza (a nickname, keep) |
  | Svobodný přístav | Free Haven (the project) |
  | Dům Svobodného přístavu | Free Haven House |
  | Studio Svobodného přístavu | Free Haven Studio (a talk show) |
  | Ježek bez klece | Ježek bez klece (a free school, keep the name) |
  | anarchokapitalismus, ankap | anarcho-capitalism, ancap |
  | anarchoagorismus | anarcho-agorism |
  | sebeřízené vzdělávání | self-directed education |
  | Svoboda učení | Freedom to Learn |
  | etatista | statist |

## Checks before sending back

- every `<video_id>.json` has a `translations/<video_id>.json` with the same number of entries (for `translate: true`)
- ids match one to one
- no Czech left in `en` (except names)
