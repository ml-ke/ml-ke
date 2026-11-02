---
title: "The Chunk Boundary Is the Bug: Measuring What a Split Costs Your Retrieval"
date: 2026-11-02 00:00:00 +0300
categories: [AI Engineering, Machine Learning]
tags: [rag, chunking, retrieval evaluation, information retrieval, bm25, text splitting]
image:
  path: /assets/img/cover-chunk-boundary-damage-rag.webp
  alt: A document column cut by two dashed chunk seams, one sentence line sliced in half across a seam, beside bars showing figure sentences kept whole rising from 54 percent at 400 characters to 100 percent with structure protection
---

> **The short version**
> A chunk is the only unit a retriever can return and a model can read. When a sentence, a table or a fenced block straddles a boundary, nothing downstream puts it back together. On this blog's own 161 posts, fixed 400-character chunks kept 54.0% of figure-bearing sentences and 19.8% of fenced blocks whole; at 800 characters, the 200-token size Chroma measured as best, the figures were 76.9% and 48.8%. Protecting fences and tables and packing the gaps at sentence boundaries kept 100% of all three, and answered 97.5% of 200 lexical probes against 54.5% for the 400-character baseline.
{: .prompt-info }

Retrieval-augmented generation has three gates. The post on [recall@k denominators](/posts/rag-recall-at-k-denominator/) covered the last one: whether the right passage appears in the top k, and whether the fraction you printed has an honest denominator. The [rank-fusion post](/posts/hybrid-retrieval-rank-fusion/) covered the middle one: how to merge two ranked lists without letting raw scores decide. This post is about the gate in front of both, the one that runs once at ingest and never again.

Chunking decides what a passage is. If two relevant sentences land in different chunks, no fusion rule and no reranker recovers the unit, because neither stage can return text it was never given. Chroma measured the effect at token level and found a spread of up to 9% in recall between strategies. Anthropic's contextual retrieval work lists chunk boundaries as an implementation consideration before it lists the embedding model. Both describe the same arithmetic: what the splitter cuts, retrieval can only hope to reassemble by luck.

## What the published evidence already shows

The [Chroma technical report](https://research.trychroma.com/evaluating-chunking) from July 2024 is the most direct measurement available. Smith and Troynikov built a token-level evaluation, because whole-document benchmarks cannot see chunking at all, and compared splitters on the same corpora. Three findings shape this post.

The spread is real but not enormous: "some strategies outperforming others by up to 9% in recall". The popular recursive character splitter does well when configured small: "the heuristic RecursiveCharacterTextSplitter with chunk size 200 and no overlap performs well" in tokens, and the report notes it "outperforms TokenTextSplitter across all metrics for chunk size of 400 or less with no chunk overlap". And the defaults can hurt. OpenAI's Assistants file search documents 800-token chunks with 400 tokens of overlap, and the report measures that setting as "slightly below-average recall and the lowest scores across all other metrics".

The library default is a separate problem. LangChain's base splitter starts at `chunk_size=4000` with `chunk_overlap=200`, and those are characters, not tokens, so the out-of-the-box setting is roughly a thousand tokens, five times the size Chroma found best. Anyone who calls the splitter without arguments is running the setting that report ranked near the bottom.

[Anthropic's contextual retrieval post](https://www.anthropic.com/news/contextual-retrieval) reports larger gains from a different intervention, prepending a short document-level context to each chunk before embedding it: a 35% reduction in top-20 retrieval failures from contextual embeddings alone (5.7% to 3.7%), 49% combined with contextual BM25 (5.7% to 2.9%), and 67% once a reranker is added. Before any of that, under implementation considerations, the post puts boundaries first: "the choice of chunk size, chunk boundary, and chunk overlap can affect retrieval performance".

[Lost in the Middle](https://arxiv.org/abs/2307.03172) adds the reason a cut is expensive rather than merely untidy. Liu and colleagues found that "performance is often highest when relevant information occurs at the beginning or end of the input context, and significantly degrades when models must access relevant information in the middle of long contexts". Half a sentence at the seam of a retrieved chunk sits in the worst place a fact can sit.

Pinecone's guide states the operating rule in one line: "if the chunk of text makes sense without the surrounding context to a human, it will make sense to the language model as well". That is a testable property, and the rest of this post turns it into a number.

## The corpus, and what counts as an atom

The corpus is this blog: the 161 posts published before this one, 2,485,607 characters of body text with front matter stripped, 621,688 tokens under the o200k tokenizer. That works out to 4.00 characters per token, which is the bridge between the character budgets below and the token budgets the vendor docs quote. It is measured, not assumed:

{% raw %}
```python
import glob, re, tiktoken

SELF = "_posts/2026-11-02-chunk-boundary-damage-rag.md"
enc = tiktoken.get_encoding("o200k_base")
chars = tokens = 0
for p in sorted(glob.glob("_posts/*.md")):
    if p == SELF:
        continue
    text = re.sub(r"^---\n.*?\n---\n", "", open(p, encoding="utf-8").read(), flags=re.S)
    chars += len(text)
    tokens += len(enc.encode(text))
print(f"{chars:,} characters, {tokens:,} tokens, {chars / tokens:.2f} characters per token")
print(f"an 800-character chunk is {800 * tokens / chars:.0f} tokens")
```
{% endraw %}

```text
2,485,607 characters, 621,688 tokens, 4.00 characters per token
an 800-character chunk is 200 tokens
```

So an 800-character budget is the 200-token size Chroma found best, which makes the comparison below a fair test of the size rather than a strawman. What varies is where the cut lands.

The unit of damage is an atom: a piece of text that has to arrive whole to be usable. This corpus has four kinds a reader can label mechanically.

- **Fenced blocks**, opener through closer. A block split at a fence line is two fragments that neither parse nor run.
- **Tables**, the whole contiguous run of rows, because the header decides what every row means.
- **Rows**, because a row whose header lives in another chunk has values but no labels.
- **Figure sentences**, prose sentences of ten words or more that contain a digit. These are the evidence units a question usually targets, and they become the query set later.

The macro structure of the corpus is 696 fenced blocks, 315 tables and 2,362 rows, with 3,206 figure sentences and 968 prose sentences that carry an inline link. Each class is small enough to label with a regex and a line scanner, which is the reason the measurement can run on any markdown corpus rather than on an annotated benchmark.

## Measure what the split destroys

The script builds seven chunker configurations over the same text, returns character spans rather than strings so containment is exact, and counts an atom as intact only when one chunk's span contains the whole atom. No fuzzy matching, no overlap heuristics, no embedding model.

{% raw %}
```python
import glob, math, random, re
from collections import Counter, defaultdict

ROW = re.compile(r"^\|.*\|[ \t]*$")
FENCE = "`" * 3

SELF = "_posts/2026-11-02-chunk-boundary-damage-rag.md"   # the corpus is the posts before this one

def corpus():
    docs = []
    for p in sorted(glob.glob("_posts/*.md")):
        if p == SELF:
            continue
        t = open(p, encoding="utf-8").read()
        docs.append((p, re.sub(r"^---\n.*?\n---\n", "", t, flags=re.S)))
    return docs

def paragraphs(text):
    out, pos = [], 0
    for m in re.finditer(r"\n[ \t]*\n", text):
        if m.start() > pos:
            out.append((pos, m.start()))
        pos = m.end()
    if pos < len(text):
        out.append((pos, len(text)))
    return out

def sentences(text):
    out = []
    for ps, pe in paragraphs(text):
        para = text[ps:pe]
        cut = [0] + [m.end() for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(])', para)]
        bounds = sorted(set(cut)) + [len(para)]
        for a, b in zip(bounds, bounds[1:]):
            seg = para[a:b]
            if seg.strip():
                lead = len(seg) - len(seg.lstrip())
                out.append((ps + a + lead, ps + a + lead + len(seg.strip())))
    return out

def fences(text):
    spans, start, i = [], None, 0
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith(FENCE):
            if start is None:
                start = i
            else:
                spans.append((start, i + len(line)))
                start = None
        i += len(line)
    return spans

def tables(text):
    spans, start, i, count = [], None, 0, 0
    for line in text.splitlines(keepends=True):
        if ROW.match(line.rstrip("\n")):
            if start is None:
                start, count = i, 0
            count += 1
        else:
            if start is not None and count >= 2:
                spans.append((start, i))
            start = None
        i += len(line)
    if start is not None and count >= 2:
        spans.append((start, len(text)))
    return spans

def atoms(text):
    fence, table = fences(text), tables(text)
    def inside(s, spans):
        return any(a < s[1] and s[0] < b for a, b in spans)
    prose = [s for s in sentences(text) if not inside(s, fence) and not inside(s, table)]
    return {"code": fence,
            "table": table,
            "row": [(m.start(), m.end()) for m in re.finditer(r"(?m)^\|[^\n]*\|[ \t]*$", text)],
            "num": [s for s in prose if len(text[s[0]:s[1]].split()) >= 10
                    and re.search(r"\d", text[s[0]:s[1]])],
            "cite": [s for s in prose if "](" in text[s[0]:s[1]]]}

def spans_fixed(text, size, overlap=0):
    step = max(1, size - overlap)
    return [(i, min(i + size, len(text))) for i in range(0, len(text), step)]

def pack(pieces, size):
    out, cur = [], None
    for a, b in pieces:
        while b - a > size:
            if cur is not None:
                out.append(tuple(cur)); cur = None
            out.append((a, a + size)); a += size
        if cur is None:
            cur = [a, b]
        elif b - cur[0] <= size:
            cur[1] = b
        else:
            out.append(tuple(cur)); cur = [a, b]
    if cur is not None:
        out.append(tuple(cur))
    return out

SEPS = ["\n\n", "\n", ". ", " "]

def split_recursive(text, a, b, seps, size):
    if b - a <= size or not seps:
        return [(a, b)]
    sep, rest = seps[0], seps[1:]
    if sep == "":
        return [(i, min(i + size, b)) for i in range(a, b, size)]
    cuts, pos = [], a
    for m in re.finditer(re.escape(sep), text[a:b]):
        cuts.append((pos, a + m.end())); pos = a + m.end()
    if pos < b:
        cuts.append((pos, b))
    out = []
    for x, y in cuts:
        out += split_recursive(text, x, y, rest, size)
    return out

def gap_pieces(spans, a, b):
    out, pos = [], a
    for s in spans:
        if s[0] >= a and s[1] <= b:
            if s[0] > pos:
                out.append((pos, s[0]))
            out.append(s)
            pos = s[1]
    if pos < b:
        out.append((pos, b))
    return out

def spans_paragraph(text, size):
    return pack(paragraphs(text), size)

def spans_sentence(text, size):
    return pack(sentences(text), size)

def spans_recursive(text, size):
    return pack(split_recursive(text, 0, len(text), SEPS, size), size)

def spans_structured(text, size):
    hard = sorted(fences(text) + tables(text))
    sents = sentences(text)
    pieces, pos = [], 0
    for a, b in hard:
        if a > pos:
            pieces += gap_pieces(sents, pos, a)
        pieces.append((a, min(b, len(text)))); pos = max(pos, b)
    if pos < len(text):
        pieces += gap_pieces(sents, pos, len(text))
    return pack(pieces, size)

CHUNKERS = {"fixed 400": lambda t: spans_fixed(t, 400),
            "fixed 800": lambda t: spans_fixed(t, 800),
            "fixed 800/120": lambda t: spans_fixed(t, 800, 120),
            "paragraph 800": lambda t: spans_paragraph(t, 800),
            "recursive 800": lambda t: spans_recursive(t, 800),
            "sentence 800": lambda t: spans_sentence(t, 800),
            "structured 800": lambda t: spans_structured(t, 800)}

def pct(values, q):
    v = sorted(values); k = (len(v) - 1) * q / 100
    lo = int(k); hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)

docs = corpus()
total = sum(len(t) for _, t in docs)
counts = Counter()
for _, t in docs:
    for k, v in atoms(t).items():
        counts[k] += len(v)
print(f"corpus   {len(docs)} posts, {total:,} characters")
print(f"atoms    {counts['code']} fenced blocks, {counts['table']} tables, {counts['row']} rows, "
      f"{counts['num']} figure sentences, {counts['cite']} citing sentences")

print(f"\natom length in characters      p50      p90      p99      max")
for k, label in (("code", "fenced block"), ("table", "table"), ("row", "table row"), ("num", "figure sentence")):
    L = [s[1] - s[0] for _, t in docs for s in atoms(t)[k]]
    print(f"  {label:16s}{pct(L, 50):10.0f}{pct(L, 90):9.0f}{pct(L, 99):9.0f}{max(L):9d}")

print(f"\na. splitter comparison, 800-character budget, atoms of 800 characters or less")
print(f"{'chunker':16s}{'chunks':>7s}{'mean':>6s}{'code':>7s}{'table':>7s}{'row':>7s}{'num':>7s}{'cite':>7s}{'dup':>6s}")
for name, fn in CHUNKERS.items():
    per = {k: [0, 0] for k in ("code", "table", "row", "num", "cite")}
    n = chars = 0
    for _, t in docs:
        ch = fn(t)
        n += len(ch); chars += sum(b - a for a, b in ch)
        for k, spans in atoms(t).items():
            for s in spans:
                if s[1] - s[0] <= 800:
                    per[k][1] += 1
                    per[k][0] += any(a <= s[0] and s[1] <= b for a, b in ch)
    print(f"{name:16s}{n:7d}{chars / n:6.0f}"
          + "".join(f"{per[k][0] / per[k][1]:7.1%}" for k in ("code", "table", "row", "num", "cite"))
          + f"{chars / total:6.2f}")

queries = []
for p, t in docs:
    for s in atoms(t)["num"]:
        sent = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t[s[0]:s[1]])
        words = [w.lower() for w in re.findall(r"[A-Za-z][A-Za-z.+-]+", sent) if len(w) >= 6]
        if len(words) >= 3:
            queries.append((p, s, " ".join(sorted(set(words), key=lambda w: (-len(w), w))[:3])))
random.Random(11).shuffle(queries)
queries = queries[:200]

def evaluate(fn):
    chunks, owner = [], []
    for p, t in docs:
        for a, b in fn(t):
            chunks.append(t[a:b]); owner.append((p, a, b))
    idx, lens, df = defaultdict(list), [], Counter()
    for i, ch in enumerate(chunks):
        tf = Counter(w.lower() for w in re.findall(r"[A-Za-z0-9][A-Za-z0-9.+-]*", ch))
        lens.append(sum(tf.values()))
        for w, c in tf.items():
            idx[w].append((i, c)); df[w] += 1
    avg, N = sum(lens) / len(lens), len(chunks)
    hit = cut = ctx = 0
    for p, s, q in queries:
        agg = Counter()
        for w in q.split():
            if w in idx:
                idf = math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5))
                for j, c in idx[w]:
                    agg[j] += idf * c * 2.2 / (c + 1.2 * (1 - 0.75 + 0.75 * lens[j] / avg))
        top = [j for j, _ in agg.most_common(5)]
        ctx += sum(len(chunks[j]) for j in top)
        whole = any(owner[j][0] == p and owner[j][1] <= s[0] and s[1] <= owner[j][2] for j in top)
        touch = any(owner[j][0] == p and not (owner[j][2] <= s[0] or owner[j][1] >= s[1]) for j in top)
        hit += whole
        cut += (not whole) and touch
    return hit / len(queries), cut / len(queries), ctx / len(queries)

print(f"\nb. {len(queries)} lexical queries, three longest words of a figure sentence")
print(f"{'chunker':16s}{'answer@5':>9s}{'cut':>7s}{'ctx chars':>10s}{'ctx tokens':>11s}")
for name, fn in CHUNKERS.items():
    h, c, x = evaluate(fn)
    print(f"{name:16s}{h:9.1%}{c:7.1%}{x:10.0f}{round(x / 4):11d}")

print(f"\nc. structure-aware splitter, five budgets, every atom counted")
print(f"{'budget':>7s}{'chunks':>7s}{'mean':>6s}{'code':>7s}{'table':>7s}{'num':>7s}{'ctx tokens':>11s}")
for size in (400, 800, 1200, 1600, 2400):
    fn = lambda t, s=size: spans_structured(t, s)
    per = {k: [0, 0] for k in ("code", "table", "num")}
    n = chars = 0
    for _, t in docs:
        ch = fn(t); n += len(ch); chars += sum(b - a for a, b in ch)
        for k in per:
            for s in atoms(t)[k]:
                per[k][1] += 1
                per[k][0] += any(a <= s[0] and s[1] <= b for a, b in ch)
    h, c, x = evaluate(fn)
    print(f"{size:7d}{n:7d}{chars / n:6.0f}" + "".join(f"{per[k][0] / per[k][1]:7.1%}" for k in ("code", "table", "num")) + f"{round(x / 4):11d}")

def header_span(text, heads, a, b):
    for ta, tb in heads:
        if ta <= a < tb or a <= ta < b:
            end = text.find("\n", ta)
            return (ta, len(text) if end < 0 else end + 1)
    return None

carry = tot = cont_rows = cont_chunks = 0
for _, t in docs:
    heads = tables(t)
    for a, b in spans_structured(t, 800):
        hs = header_span(t, heads, a, b)
        rows = [r for r in atoms(t)["row"] if a <= r[0] and r[1] <= b]
        tot += len(rows)
        if hs and a <= hs[0] and hs[1] <= b:
            carry += len(rows)
        elif hs and hs[0] < a:                       # chunk starts inside a table
            cont_chunks += 1
            cont_rows += len(rows)
print(f"\nd. rows in a chunk that also holds their header row: {carry}/{tot} = {carry / tot:.1%}")
print(f"   rows in a chunk that begins mid-table: {cont_rows}/{tot} = {cont_rows / tot:.1%}"
      f" across {cont_chunks} chunks, none carrying a header")
```
{% endraw %}

```text
corpus   161 posts, 2,485,607 characters
atoms    696 fenced blocks, 315 tables, 2362 rows, 3206 figure sentences, 968 citing sentences

atom length in characters      p50      p90      p99      max
  fenced block           663     2548     5836    15517
  table                  580     1068     1648     2102
  table row               78      157      242      668
  figure sentence        169      398     1472     3107

a. splitter comparison, 800-character budget, atoms of 800 characters or less
chunker          chunks  mean   code  table    row    num   cite   dup
fixed 400          6294   395  19.8%   8.0%  78.1%  54.0%  30.3%  1.00
fixed 800          3189   779  48.8%  35.0%  88.8%  76.9%  63.1%  1.00
fixed 800/120      3731   781  58.8%  44.2%  98.6%  87.5%  70.3%  1.17
paragraph 800      3983   622  53.5%  57.1%  96.2%  95.6%  91.1%  1.00
recursive 800      3915   635  82.0% 100.0% 100.0%  97.1%  93.2%  1.00
sentence 800       3748   661  62.7%  80.1%  96.0% 100.0% 100.0%  1.00
structured 800     3736   665 100.0% 100.0%  96.2% 100.0% 100.0%  1.00

b. 200 lexical queries, three longest words of a figure sentence
chunker          answer@5    cut ctx chars ctx tokens
fixed 400           54.5%  43.0%      1937        484
fixed 800           78.5%  19.5%      3816        954
fixed 800/120       86.0%  13.0%      3818        955
paragraph 800       92.5%   6.5%      3032        758
recursive 800       95.5%   3.5%      3068        767
sentence 800        97.5%   1.0%      3237        809
structured 800      97.5%   1.0%      3206        801

c. structure-aware splitter, five budgets, every atom counted
 budget chunks  mean   code  table    num ctx tokens
    400   7545   329  30.3%  17.1%  90.1%        383
    800   3736   665  57.5%  71.7%  97.5%        801
   1200   2482  1001  71.3%  94.3%  98.5%       1205
   1600   1867  1331  79.9%  98.7%  99.2%       1646
   2400   1254  1982  89.1% 100.0%  99.9%       2441

d. rows in a chunk that also holds their header row: 2161/2272 = 95.1%
   rows in a chunk that begins mid-table: 111/2272 = 4.9% across 93 chunks, none carrying a header
```

Read the first table as four verdicts, not one ranking.

The baseline is a 400-character fixed window, roughly 100 tokens, the size a first-pass script reaches for. It keeps 19.8% of fenced blocks, 8.0% of tables and 54.0% of figure sentences. Nearly half the facts in this corpus arrive at the model as fragments, and four out of five code blocks cannot be reassembled from any single chunk.

Doubling to 800 characters, the 200-token size Chroma measured as best, improves every column and still leaves 23.1% of the prose evidence cut. Fixed windows pay for their simplicity in exactly the currency this post is counting.

Overlap is the first repair available, and it works. 800 characters with 120 characters of overlap raises fence survival from 48.8% to 58.8% and figure sentences from 76.9% to 87.5%. It also re-embeds 1.17 characters for every character in the corpus, a 17% bill on ingest and storage that buys about ten points on one column.

Structure-aware splitters do better at the same budget without duplication. The recursive splitter, which tries paragraph breaks before line breaks before sentence-ending punctuation, keeps 82.0% of fenced blocks, 100% of tables and 97.1% of figure sentences. Sentence packing reaches 100% on figure sentences and citing sentences, because it cannot cut mid-sentence, but it drops to 62.7% on fences and 80.1% on tables: a splitter that only knows about full stops will happily bisect a code block.

The structured splitter in the last row is the one built for this post. Fences and whole tables are indivisible pieces, and only the gaps between them are packed at sentence boundaries. It keeps 100% of blocks, 100% of tables and 100% of figure sentences, at 1.00 duplication and 3,736 chunks against 3,748 for plain sentence packing. The remaining weak spot is the row column at 96.2%: tables longer than 800 characters get sliced, and a sliced table loses its header.

## The damage shows up in the answer

Survival is a property of the index. The number that matters is whether the evidence reaches the model when someone asks, so the second experiment builds 200 lexical queries from figure sentences, three longest content words each, and scores every chunk with BM25 before taking the top five.

BM25 stands in for dense retrieval because it runs in the standard library and shares the relevant geometry: both return whole chunks, so both inherit whatever the splitter cut. The query is deliberately easy, drawn from the sentence itself, so the test measures boundary damage rather than question difficulty. The `cut` column isolates the interesting failure, a top-five chunk that overlaps the evidence sentence while no chunk contains it whole.

| chunker | answer@5 | cut | context per query |
|---|---|---|---|
| fixed 400 | 54.5% | 43.0% | 484 tokens |
| fixed 800 | 78.5% | 19.5% | 954 tokens |
| fixed 800/120 | 86.0% | 13.0% | 955 tokens |
| paragraph 800 | 92.5% | 6.5% | 758 tokens |
| recursive 800 | 95.5% | 3.5% | 767 tokens |
| sentence 800 | 97.5% | 1.0% | 809 tokens |
| structured 800 | 97.5% | 1.0% | 801 tokens |

The cut rate and the miss rate move together. At 400 characters, 43.0% of queries return a chunk that overlaps the evidence without holding it, and answer@5 lands at 54.5%; the two columns account for 97.5% of the 200 queries, and the remainder is a plain rank miss where no top-five chunk touches the evidence at all. At 800 characters the pair is 19.5% and 78.5%. Sentence packing and structure protection both reach 1.0% cut and 97.5% answered. When a sentence is split, the retriever usually finds its neighbourhood, and the model still cannot read the fact.

Context cost is the column that keeps this honest. The 400-character baseline looks cheap at 484 tokens per query, but it is cheap because it usually returns half a fact. The splitters that preserve atoms deliver about 800 tokens of context per query, which is what a top-5 answer should cost when the units are intact.

Two limits apply to this experiment. It is one corpus, chosen because its readers can clone it, and the scorer is lexical where production retrieval is usually dense. The direction of the effect is what transfers, and the 43.0% cut rate at 400 characters is the number to reproduce on your own documents rather than quote as a constant.

## The budget comes first, the boundary second

Both tables above hold the budget at 800 characters and exclude atoms longer than that, because no splitter can keep a 2,548-character fence whole inside an 800-character chunk. The exclusion makes the splitter comparison fair. It also hides the first decision.

The size distribution says it plainly. The median fenced block is 663 characters and the ninetieth percentile is 2,548, with a maximum of 15,517. Tables have a median of 580 and a p90 of 1,068. Figure sentences are small, with a p90 of 398. A budget chosen from prose alone will shred the structures, and a budget chosen from the code tail will dilute the prose.

The third experiment counts every atom, including those the budget cannot hold, across five budgets for the structured splitter.

| budget (characters) | chunks | mean chunk | code | table | figure sentence | context per query |
|---|---|---|---|---|---|---|
| 400 | 7,545 | 329 | 30.3% | 17.1% | 90.1% | 383 tokens |
| 800 | 3,736 | 665 | 57.5% | 71.7% | 97.5% | 801 tokens |
| 1,200 | 2,482 | 1,001 | 71.3% | 94.3% | 98.5% | 1,205 tokens |
| 1,600 | 1,867 | 1,331 | 79.9% | 98.7% | 99.2% | 1,646 tokens |
| 2,400 | 1,254 | 1,982 | 89.1% | 100.0% | 99.9% | 2,441 tokens |

Two things move in opposite directions. Coverage of the long structures climbs with the budget, from 30.3% of code at 400 characters to 89.1% at 2,400, and so does the per-query context bill, from 383 tokens to 2,441. Prose coverage is already 97.5% at 800 characters and gains 2.4 points by 2,400, while the context bill multiplies by three. For a corpus like this one, the budget is a decision about whether code and tables count as first-class evidence; the prose comes along for the ride.

The last measurement is small and cheap to fix. At an 800-character budget, the structured splitter leaves 111 of 2,272 table rows in a chunk that begins mid-table, across 93 chunks, none of which carries the header. A row without its header is a value without its label. Prepending the header line to each table chunk repairs all 111 and adds roughly 80 characters to the affected chunks.

## How to apply this

1. **Measure the atom sizes before choosing a size.** Extract your fenced blocks, tables, rows and evidence sentences, and print p50 and p90. If the p90 of the class you care about exceeds your candidate budget, the budget is the constraint and no splitter change fixes it.
2. **Keep atoms whole.** Treat fences and tables as indivisible pieces and pack only the gaps. That is the gap between 62.7% and 100% fence survival in the first table, and it costs nothing at ingest.
3. **Count survival, not vibes.** The metric is one line: the fraction of atoms contained in a single chunk. Print it next to chunk count and mean length, and keep it as a regression test for the day someone changes the splitter or the document templates.
4. **Separate cut from rank-miss.** A failure where the evidence sat in a top-k chunk but split is a preprocessing bug and belongs to the splitter's owner. A failure where the evidence chunk was never retrieved is a ranking or embedding problem. Mixing them sends the fix to the wrong team.
5. **Buy overlap last.** 120 characters of overlap on 800-character windows bought about ten points of survival for 17% duplicated ingest. Structure protection bought more for free, so spend the duplication budget after the boundary work is done.
6. **Attach context to the chunk.** The Anthropic result is about giving a chunk a sentence of document-level context before embedding; the cheap local version is a heading path, plus the repeated header row for tables. Both are string operations at index time and neither changes the split.

## Key takeaways

| Question | Measured answer |
|---|---|
| What does a 400-character fixed window destroy? | 80.2% of fenced blocks, 92.0% of tables, 46.0% of figure sentences, and 45.5 points of answer@5 |
| Does doubling the window fix it? | No. At 800 characters, 23.1% of figure sentences and 51.2% of fenced blocks are still cut |
| What does structure protection cost? | Nothing at ingest: 1.00 duplication, 3,736 chunks against 3,748, and 100% of blocks, tables and figure sentences intact |
| Is the size or the boundary the bigger decision? | Size first: at 800 characters 57.5% of all code survives, at 2,400 it is 89.1%, whatever the splitter |
| What should be monitored in production? | Atom survival per corpus, re-measured whenever the splitter, templates or document mix change |

## References

- Smith, B. and Troynikov, A., *Evaluating Chunking Strategies for Retrieval*, Chroma Technical Report, July 3, 2024 — https://research.trychroma.com/evaluating-chunking
- Anthropic, *Introducing Contextual Retrieval*, September 19, 2024 — https://www.anthropic.com/news/contextual-retrieval
- Liu, N. F. et al., *Lost in the Middle: How Language Models Use Long Contexts*, TACL 2023 — https://arxiv.org/abs/2307.03172
- LangChain, `langchain_text_splitters` base splitter, defaults `chunk_size=4000`, `chunk_overlap=200`, `length_function=len` — https://github.com/langchain-ai/langchain/blob/master/libs/text-splitters/langchain_text_splitters/base.py
- Pinecone, *Chunking Strategies for LLM Applications*, June 28, 2025 — https://www.pinecone.io/learn/chunking-strategies/
- OpenAI, `tiktoken` o200k_base encoding, used for the characters-per-token measurement — https://github.com/openai/tiktoken

## Related posts

- [The Denominator Is the Metric: Auditing a RAG Retriever Before You Blame the Model](/posts/rag-recall-at-k-denominator/) — the ranking layer this post feeds
- [Hybrid Retrieval and Rank Fusion](/posts/hybrid-retrieval-rank-fusion/) — what happens to the chunks once they exist
- [Filtered Vector Search Pool Sizing](/posts/filtered-vector-search-pool-sizing/) — how metadata filters shrink the candidate pool
- [RAG for Low-Resource African Languages](/posts/rag-low-resource-african-languages/) — chunking when the tokenizer is the constraint
