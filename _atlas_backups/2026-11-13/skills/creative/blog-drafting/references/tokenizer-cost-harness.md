# Tokenizer-Cost Harness (tokens per byte across vocabularies)

Added Oct 8 2026 while writing the MORENA post. Use this instead of re-deriving the
measurement whenever a post needs to show what a tokenizer costs on a non-English
language. No GPU, one small download, ~45 s wall clock.

## Why it is the right demo for small-vocabulary model releases

Multilingual model releases are usually covered via benchmark tables (bits per byte,
chrF++). Those numbers are developer-published and un-verifiable by a reader. The
*mechanism* — that an English-centric vocabulary forces non-English text into more,
shorter tokens, so the same sentence costs more money and eats more context — is
arithmetic anyone can reproduce. Measure it yourself and the post has a claim with a
tool attached rather than a number quoted from a press release.

## Recipe (verified)

```bash
uv run --with=tokenizers --with=tiktoken python3 tokenizer_cost.py
```

- Subject tokenizer: `https://huggingface.co/<org>/<model>/resolve/main/tokenizer.json`
  (4.7 MB for MORENA) → `Tokenizer.from_str(urllib.request.urlopen(req).read().decode())`.
  **`Tokenizer.from_file` takes a path string, not a BytesIO** — `from_str` avoids a temp file.
- Reference tokenizers: `tiktoken.get_encoding("o200k_base")` (GPT-4o) and
  `"cl100k_base"` (GPT-4). Both download their BPE file at runtime.
- Metric: `tokens / len(text.encode("utf-8"))` — tokens per **byte**, not per character,
  so multi-byte diacritics are counted honestly.
- Publish two tables: absolute tpB per language, then the **penalty ratio**
  (`tpB(lang) / tpB(English)`) per tokenizer. The ratio is the finding; the absolute
  counts mislead because a 65k vocabulary legitimately spends more tokens on English.
- Freeze the sample texts as string literals in the block. Do not fetch Wikipedia at
  run time — an edit changes the bytes and the quoted output stops reproducing.

## Verified fixture numbers — MORENA vs GPT-4o vs GPT-4 (Oct 8 2026)

Vocab sizes: MORENA 65,536 | o200k 200,019 | cl100k 100,277.

| language | bytes | MORENA tpB | o200k tpB | cl100k tpB |
|---|---|---|---|---|
| English | 99 | 0.232 | 0.212 | 0.212 |
| Swahili | 96 | 0.240 | 0.281 | 0.396 |
| Hausa | 66 | 0.227 | 0.303 | 0.379 |
| Yoruba | 148 | 0.223 | 0.405 | 0.534 |

Penalty vs English: Swahili 1.03 / 1.33 / 1.87; Hausa 0.98 / 1.43 / 1.79;
Yoruba 0.96 / 1.91 / 2.52. Means across the three African samples:
**MORENA 0.99x, o200k 1.56x, cl100k 2.06x.**

Determinism reference: sha256 of the block's stdout = `1eba253c3b3f103d0023fd8339c2a60268147b4f7437ae7f3806a8040cbf2f05`
(identical across reruns; greedy tokenization is deterministic, so publish the hash when
the post leans on the exact figures).

Sample texts were the first sentences of the Wikipedia summaries for Nairobi
(en + sw) and Nigeria (`Najeriya` ha, `Nàìjíríà` yo), fetched 2026-10-08. Wikipedia
REST summary API: `https://<lang>.wikipedia.org/api/rest_v1/page/summary/<title>`
— URL-encode non-ASCII titles (`urllib.parse.quote`), or the request fails with
`'ascii' codec can't encode` / `URL can't contain control characters`.

Derived figure worth reusing: at 0.240 vs 0.396 tpB, a 4,096-token window holds
~17,067 bytes of Swahili under MORENA's tokenizer against ~10,343 under cl100k
(1.65x more document in the same window).

## Traps

- **Do not merge the base and instruct figures.** A vendor's headline bits-per-byte is
  usually the *base* checkpoint; the instruct sibling measures higher. MORENA base 1.408
  vs instruct 1.441 — TechRadar quoted 1.408, the instruct card says 1.441. Name the
  checkpoint in the sentence.
- **The subject model can lose on its own English text.** MORENA's 0.232 tpB on English
  is worse than o200k's 0.212. Say so — it is the expected trade for a small vocabulary
  and pre-empting it is more credible than hiding it.
- **`o200k_base` is not available in some tiktoken versions** — verify both encodings
  resolve before publishing, and state which two references you compared.
- **Gemma and Llama tokenizers are gated.** Do not promise a comparison against Gemma 3
  or Llama 3.2 unless you have HF auth; compare against tiktoken instead and say so.
- Attribute vendor tokenizer claims to the vendor (`1.39x fewer tokens than Gemma 3`,
  `1.53x fewer than Llama 3.2`) and keep them distinct from your own measurement.
