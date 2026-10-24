---
title: "Fuse Ranks, Not Scores: Combining BM25 and Dense Retrieval"
date: 2026-10-24 00:00:00 +0300
categories: [Machine Learning, AI Engineering]
tags: [hybrid search, reciprocal rank fusion, BM25, information retrieval, RAG, evaluation]
image:
  path: /assets/img/cover-hybrid-retrieval-rank-fusion.webp
  alt: Two ranked lists, one lexical and one semantic, merged by reciprocal rank into a single list
---

> **The short version**
> Hybrid retrieval works. The way most teams combine the two lists quietly throws away half the signal. This post builds a 61-document support corpus with a BM25 index and a semantic index, scores four fusion strategies on 30 labelled queries, and reports where each one wins and loses. Every number below comes from the code in this post, run on this laptop, and both blocks re-run byte-identical.
{: .prompt-info }

Retrieval-augmented systems rarely fail because the embedding model is bad. They fail because two retrievers return two lists and nobody decided how to merge them. Recall at ten looks fine, so the merge gets one line of arithmetic and no evaluation.

That line of arithmetic matters more than the embedding model choice, for two reasons. Lexical and semantic retrievers fail on different queries, so the union of their candidates carries information that neither list has alone. And their scores live on scales you cannot compare, so the obvious merge (add the two scores) degrades into "use one retriever and keep the other as a tiebreaker", without raising an error.

This post takes that apart with runnable code. If you are here for how to measure one retriever rather than how to combine two, [the recall denominator post](/posts/rag-recall-at-k-denominator/) and [the filtered ANN pool sizing post](/posts/filtered-vector-search-pool-sizing/) cover the measurement side.

## Two retrievers, two failure modes

The demo corpus is a farm-advisory knowledge base: 61 short documents, each with a title, a body, and a set of ground-truth concepts. Queries come in three flavours, because real traffic does too.

| Query class | Example | What it needs |
|---|---|---|
| Literal | `agri-4021`, `kephis`, `kalro` | Exact token match. A product code has no semantic neighbourhood. |
| Paraphrase | "a way to grow plants without rain" | Meaning match. The document says "drip irrigation scheduling" and shares no content word with the query. |
| Mixed | "drip lines iron" | Both. One anchor token plus a paraphrase hint. |

The lexical side is BM25 with the Lucene defaults, `k1 = 1.2` and `b = 0.75`, and the idf term `log(1 + (docCount - docFreq + 0.5) / (docFreq + 0.5))`, which is exactly [what Lucene's `BM25Similarity` computes](https://lucene.apache.org/core/9_10_0/core/org/apache/lucene/search/similarities/BM25Similarity.html).

The semantic side is a stand-in: it scores each document by concept overlap with the query's intended meaning, plus a seeded noise floor so it is semantic but not perfect. It stands in for an embedding index on purpose, since the point here is the fusion arithmetic, and a stand-in makes every number reproducible without downloading a model. Swap in your own vectors and the arithmetic in the rest of the post is unchanged; your absolute scores will differ.

{% raw %}
```python
import math, random
from collections import Counter

# One line per document: title :: body :: ground-truth concepts
RAW = """
maize fall armyworm scouting :: scout maize fields weekly for fall armyworm; check whorls for window pane damage and frass :: pest maize scouting
agri-4021 fertiliser blend :: top dress maize with AGRI-4021 blend at six weeks; apply 50 kg per acre near the stem :: fertiliser maize topdress
kephis seed certification :: buy certified seed carrying the KEPHIS label; unlabelled seed carries germination risk :: seed certification regulation
drip irrigation scheduling :: run drip lines early morning; umwagiliaji twice a week beats daily light watering :: irrigation water scheduling
soil ph testing :: test soil ph yearly; maize tolerates 5.5 to 7.0, add lime when below 5.5 :: soil ph amendment
post harvest storage :: dry grain to 13 percent moisture before storage; use hermetic bags against weevils :: storage postharvest moisture
aphid outbreak on beans :: aphids cluster under bean leaves; spray neem extract at dusk to protect pollinators :: pest beans spraying
kalro variety trial results :: KALRO trials show drought tolerant variety yields 2.1 tonnes per hectare under short rains :: variety drought yield
mbolea application timing :: apply mbolea at planting rather than after emergence so roots reach nutrients early :: fertiliser timing planting
wadudu monitoring traps :: hang yellow sticky traps at canopy height to count wadudu before spraying :: pest monitoring traps
coffee berry borer :: prune shade trees and strip fallen berries to break coffee berry borer cycle :: pest coffee pruning
rainfall onset forecast :: onset forecasts from the county met office guide planting dates; do not plant on a single wet day :: weather planting forecast
greenhouse ventilation :: open side vents by nine am; leaf curl signals heat stress inside the tunnel :: greenhouse heat ventilation
afa registration :: food exporters must register with AFA and keep traceability records per consignment :: regulation export traceability
irrigation water testing :: test irrigation water for salinity; above 2 ds per metre switch to tolerant crops :: water salinity testing
maize lethal necrosis :: rogue out plants with yellowing and dead heart; maize lethal necrosis spreads by thrips :: disease maize rogueing
tissue culture bananas :: tissue culture banana plantlets are disease free; harden them in shade for two weeks :: banana propagation seedling
compost heap layering :: layer dry stalks and green matter, turn every three weeks, keep heaps moist but not soaked :: soil compost organic
avocado export grade :: grade avocados by diameter; exporters reject fruit below 55 millimetres :: export avocado grading
poultry feed conversion :: track feed conversion weekly; a rise above 1.8 signals disease or poor ventilation :: poultry feed monitoring
agri-7733 foliar spray :: spray AGRI-7733 foliar at flowering only; later sprays leave residue above the export limit :: fertiliser spraying residue
storage pest fumigation :: fumigate silos under licensed supervision; phosphine needs seven days sealed :: storage pest fumigation
drought tolerant sorghum :: sorghum out yields maize in marginal rainfall; plant at onset on 75 centimetre rows :: variety drought sorghum
water pan construction :: line a farm water pan with clay and compact the floor to cut seepage losses :: water storage construction
leaf spot on kale :: remove lower kale leaves showing concentric spots and rotate beds for two seasons :: disease kale rotation
extension officer visits :: group requests for extension visits go through the ward agriculture office each quarter :: extension access services
mobile advisory sms :: subscribe to the county sms advisory; messages carry planting windows and pest alerts :: advisory sms access
soil erosion terraces :: cut terraces on slopes above eight percent to hold topsoil through the long rains :: soil erosion terracing
nursery shading :: shade vegetable nurseries at fifty percent until seedlings have four true leaves :: nursery shading seedling
milk cooling chain :: cool milk to four degrees within two hours; reject milk with added water :: dairy cooling quality
agri-1180 soil conditioner :: work AGRI-1180 conditioner into the top ten centimetres before planting potatoes :: soil amendment potato
fallow rotation legumes :: rotate cereals with legumes to rebuild nitrogen without extra fertiliser spend :: rotation legumes soil
irrigation pump maintenance :: clean pump filters weekly; a clogged filter cuts pressure and burns the motor :: irrigation maintenance equipment
market price information :: check daily wholesale prices at the county market board before harvesting for sale :: market price selling
seed rate calculation :: maize seed rate of ten kilograms per acre suits seventy five centimetre spacing :: seed spacing planting
livestock vaccination :: vaccinate cattle against east coast fever before the rains; keep the certificate :: livestock vaccination health
solar dryer fruit :: solar dry mango slices at fifty five degrees for two days to hold colour :: postharvest solar fruit
pest scouting records :: keep scouting records per plot; spray only when counts cross the action threshold :: pest records threshold
agri-5502 herbicide :: apply AGRI-5502 pre emergence within three days of planting; never on emerged maize :: herbicide weed timing
weed control mulching :: mulch with dry grass to suppress weeds and hold moisture through dry spells :: weed mulch moisture
bee keeping pollination :: place two hives per acre of avocado to lift fruit set during flowering :: pollination bees avocado
cold chain tomato :: pack tomatoes in crates not sacks; sacks crush fruit and cut market grade :: postharvest tomato packaging
salinity tolerant varieties :: where irrigation water is salty, plant salinity tolerant varieties and leach salts with rain :: water salinity variety
agronomy training days :: field days run every second thursday at the sub county showground, free to farmer groups :: extension training access
silage making :: chop forage to two centimetres, compact hard and cover to exclude air for silage :: livestock feed silage
mango fruit fly :: hang fruit fly traps and collect fallen fruit twice weekly to break the cycle :: pest mango traps
water harvesting gutter :: fit gutters to the iron roof and store runoff in a lined tank for the nursery :: water harvesting construction
agri-9081 foliar calcium :: AGRI-9081 foliar calcium prevents blossom end rot in tomatoes during fast growth :: fertiliser tomato calcium
soil carbon monitoring :: sample soil carbon every third season to track whether organic inputs are working :: soil carbon monitoring
price hedging cooperative :: cooperatives pool volumes to negotiate a forward price before harvest :: market cooperative selling
seed storage humidity :: store seed below forty percent humidity with a moisture indicator in the sack :: seed storage humidity
irrigation emitter clogging :: flush drip lines monthly; emitters clog faster with borehole water high in iron :: irrigation maintenance water
agri-6604 nematicide :: apply AGRI-6604 nematicide two weeks before tomato transplanting, not at planting :: pest nematicide tomato
livestock tick control :: spray cattle weekly in the wet season; ticks carry east coast fever :: livestock ticks spraying
post harvest loss audit :: weigh a sample after sorting to estimate post harvest loss per tonne stored :: postharvest loss records
agroforestry shade :: plant grevillea on field boundaries for timber without shading the maize rows :: agroforestry shade timber
fertiliser subsidy register :: the subsidised fertiliser register is verified at the ward office before collection :: fertiliser subsidy access
hive hygiene :: clean hives and replace old comb yearly to hold down varroa loads :: bees health maintenance
agri-2255 seed treatment :: treat seed with AGRI-2255 before planting to stop soil borne damping off :: seed treatment disease
farmer field school :: farmer field schools meet weekly through the season to compare plots side by side :: extension training access
soil test laboratory :: send composite samples to the accredited laboratory; results return in ten days :: soil testing access
"""

DOCS = []
for line in RAW.strip().splitlines():
    title, body, concepts = [p.strip() for p in line.split("::")]
    DOCS.append((title, body, set(concepts.split())))

# (query, class, gold index, intended meaning for the semantic retriever)
QUERIES = [
    ("agri-4021", "literal", 1, set()),
    ("kephis", "literal", 2, set()),
    ("agri-7733", "literal", 20, set()),
    ("agri-1180", "literal", 30, set()),
    ("agri-5502", "literal", 38, set()),
    ("agri-9081", "literal", 47, set()),
    ("agri-6604", "literal", 52, set()),
    ("agri-2255", "literal", 58, set()),
    ("kalro", "literal", 7, set()),
    ("afa registration", "literal", 13, {"regulation"}),
    ("chemicals that stop grass growing", "paraphrase", 38, {"herbicide", "weed"}),
    ("keeping grain safe from insects after drying", "paraphrase", 5, {"storage", "postharvest"}),
    ("how do i know if my land is acidic", "paraphrase", 4, {"soil", "ph"}),
    ("giving plants food when they first go in the ground", "paraphrase", 8, {"fertiliser", "planting"}),
    ("a way to grow plants without rain", "paraphrase", 3, {"irrigation", "water"}),
    ("stopping soil washing down a hill", "paraphrase", 28, {"soil", "erosion"}),
    ("cooling milk quickly after milking", "paraphrase", 29, {"dairy", "cooling"}),
    ("insect that eats maize leaves in the funnel", "paraphrase", 0, {"pest", "maize"}),
    ("buying seed that is officially approved", "paraphrase", 2, {"seed", "certification"}),
    ("keeping tomatoes from being crushed on the way to market", "paraphrase", 42, {"postharvest", "tomato"}),
    ("agri-4021 when to apply", "mixed", 1, {"fertiliser"}),
    ("aphids neem", "mixed", 6, {"pest", "beans"}),
    ("drip lines iron", "mixed", 51, {"irrigation", "maintenance"}),
    ("sorghum rows marginal rain", "mixed", 22, {"variety", "drought"}),
    ("avocado hives", "mixed", 40, {"bees", "avocado"}),
    ("silage chop length", "mixed", 45, {"livestock", "feed"}),
    ("terraces slope percent", "mixed", 28, {"soil", "erosion"}),
    ("mango traps fallen fruit", "mixed", 46, {"pest", "mango"}),
    ("tomato crates sacks", "mixed", 42, {"postharvest", "tomato"}),
    ("kale concentric spots", "mixed", 24, {"disease", "kale"}),
]

def tokenize(text):
    out = []
    for tok in text.lower().replace("-", " ").split():
        tok = "".join(ch for ch in tok if ch.isalnum())
        if tok:
            out.append(tok)
    return out

class BM25:
    """Lexical retriever, Lucene defaults: k1=1.2, b=0.75."""
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.tfs, self.lens = [], []
        df = Counter()
        for title, body, _ in docs:
            tf = Counter(tokenize(title + " " + body))
            self.tfs.append(tf)
            self.lens.append(sum(tf.values()))
            df.update(tf.keys())
        self.n = len(docs)
        self.avgdl = sum(self.lens) / self.n
        self.idf = {w: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query):
        q = tokenize(query)
        out = []
        for i, tf in enumerate(self.tfs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * (f * (self.k1 + 1)) / (f + norm)
            out.append(s)
        return out

class Dense:
    """Stand-in for an embedding retriever: matches meaning, not wording.
    Scores by concept overlap plus a seeded noise floor, so it misses sometimes."""
    def __init__(self, docs, noise=0.12, seed=11):
        self.concepts = [d[2] for d in docs]
        self.noise, self.seed = noise, seed

    def scores(self, query_concepts):
        rng = random.Random(self.seed)
        out = []
        for cs in self.concepts:
            union = len(cs | query_concepts) or 1
            out.append(len(cs & query_concepts) / union + self.noise * rng.random())
        return out

def ranked(scores, n=10):
    return [i for i, _ in sorted(enumerate(scores), key=lambda p: (-p[1], p[0]))[:n]]

def recall_at_k(rank_list, gold):
    return 1.0 if gold in rank_list else 0.0

def rr_at_k(rank_list, gold):
    for pos, doc in enumerate(rank_list, 1):
        if doc == gold:
            return 1.0 / pos
    return 0.0

bm25, dense = BM25(DOCS), Dense(DOCS)

print(f"corpus: {len(DOCS)} documents, {len(QUERIES)} labelled queries")
print(f"{'retriever':10s}{'class':12s}{'R@10':>6s}{'MRR@10':>8s}")
for name, fn in (("bm25", lambda q, c: bm25.scores(q)),
                 ("dense", lambda q, c: dense.scores(c))):
    for cls in ("literal", "paraphrase", "mixed"):
        rows = [(q, g, c) for q, k, g, c in QUERIES if k == cls]
        r = sum(recall_at_k(ranked(fn(q, c)), g) for q, g, c in rows) / len(rows)
        m = sum(rr_at_k(ranked(fn(q, c)), g) for q, g, c in rows) / len(rows)
        print(f"{name:10s}{cls:12s}{r:6.2f}{m:8.3f}")
    r = sum(recall_at_k(ranked(fn(q, c)), g) for q, _, g, c in QUERIES) / len(QUERIES)
    m = sum(rr_at_k(ranked(fn(q, c)), g) for q, _, g, c in QUERIES) / len(QUERIES)
    print(f"{name:10s}{'all':12s}{r:6.2f}{m:8.3f}")

lex_best = [max(bm25.scores(q)) for q, _, _, _ in QUERIES]
sem_best = [max(dense.scores(c)) for _, _, _, c in QUERIES]
print(f"score scale: bm25 best per query {min(lex_best):.2f}-{max(lex_best):.2f}"
      f" | dense best per query {min(sem_best):.2f}-{max(sem_best):.2f}")
```
{% endraw %}

```text
corpus: 61 documents, 30 labelled queries
retriever class         R@10  MRR@10
bm25      literal       1.00   1.000
bm25      paraphrase    0.50   0.324
bm25      mixed         0.70   0.650
bm25      all           0.73   0.658
dense     literal       0.30   0.077
dense     paraphrase    0.80   0.800
dense     mixed         0.60   0.500
dense     all           0.57   0.459
score scale: bm25 best per query 3.33-13.24 | dense best per query 0.12-0.78
```

The split is clean and it is the whole argument. BM25 gets every literal query right at rank one and half the paraphrase queries into the top ten. The semantic index gets ten of ten paraphrases near rank one and collapses on codes, at 0.30 recall and 0.077 MRR, because `agri-6604` has no meaning to embed.

Recall at ten for either retriever alone (0.73 and 0.57) is the ceiling you would report today. A fused list should beat both, and that is the thing worth measuring.

## Why adding raw scores fails quietly

The last line of output is the trap. Across the 30 queries, the weakest BM25 top score is 3.33 and the strongest is 13.24. The semantic retriever's best score for any query never exceeds 0.78.

So in `0.5 * bm25 + 0.5 * dense`, one BM25 term can outweigh the maximum possible semantic contribution by a factor of four, and often by twenty. The sum is not a 50/50 blend. It is BM25 with a tiebreaker, and the tiebreaker only fires when two documents have identical BM25 scores, which in a corpus of floating-point sums almost never happens.

That failure is invisible in aggregate. The fused system still returns ten results, still beats the semantic index alone, and any single-retriever dashboard looks healthy. It just never recovers the paraphrase queries BM25 missed.

Normalising first fixes the scale but introduces a contract you have to keep. Min-max maps each retriever's window to `[0, 1]`, which is what OpenSearch's normalisation processor does [by default](https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/), with `l2` offered for cases where "the ratios between the scores within a query clause carry information". Min-max also compresses everything else when one document is an outlier: the highest-scoring document always becomes 1.0 and its distance from the rest becomes the scale.

## What 1/(k + rank) computes

Reciprocal rank fusion ignores scores entirely and sums `1 / (k + rank)` across the lists where a document appears. Cormack, Clarke and Büttcher introduced it as a metarank baseline and found, fusing 30 configurations of one system over four TREC collections, that `k = 60` was near-optimal "but that the choice was not critical" ([SIGIR 2009, PDF](https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf)).

That sentence hides the knob. The arithmetic of `k` is the whole design decision, and it fits in three lines:

```python
def weight(rank, k):
    return 1.0 / (k + rank)

print("how much one rank is worth, by k")
for k in (0, 10, 60, 500):
    w1, w10 = weight(1, k), weight(10, k)
    print(f"  k={k:3d}  rank1={w1:.4f}  rank10={w10:.4f}  rank1/rank10={w1 / w10:5.2f}x")

print("\nrank 1 in one list  vs  rank 3 in both lists")
for k in (0, 60):
    one, both = weight(1, k), 2 * weight(3, k)
    win = "rank3-both" if both > one else "rank1-one"
    print(f"  k={k:3d}  one-list={one:.4f}  both-lists={both:.4f}  winner={win}")
```

```text
how much one rank is worth, by k
  k=  0  rank1=1.0000  rank10=0.1000  rank1/rank10=10.00x
  k= 10  rank1=0.0909  rank10=0.0500  rank1/rank10= 1.82x
  k= 60  rank1=0.0164  rank10=0.0143  rank1/rank10= 1.15x
  k=500  rank1=0.0020  rank10=0.0020  rank1/rank10= 1.02x

rank 1 in one list  vs  rank 3 in both lists
  k=  0  one-list=1.0000  both-lists=0.6667  winner=rank1-one
  k= 60  one-list=0.0164  both-lists=0.0317  winner=rank3-both
```

At `k = 60`, being ranked first by one retriever is worth barely more than being ranked tenth by it, a ratio of 1.15. A document that both retrievers place third beats a document that one retriever places first. That is the behaviour you want when the two retrievers are independent and agreement is the strongest relevance signal.

It is exactly the wrong behaviour when one retriever is simply right. On a literal query, BM25 puts the answer at rank one and the semantic index is guessing from noise. With `k = 60`, the guessing list's agreement with the other list can outvote BM25's confidence.

Elasticsearch exposes this as `rank_constant`, defaulting to 60, alongside `rank_window_size` which sets the depth of each child list. Their docs are blunt that "a higher value indicates that lower ranked documents have more influence" ([Elastic docs](https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion)). Azure AI Search describes the same formula as `1/(rank + k)` and repeats the k of 60 as the empirically good value ([Microsoft Learn](https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking)).

## Four fusion strategies, one corpus

The next block fuses the two lists four ways, sweeps a weight, and checks whether the normalisation window matters. It repeats the corpus so it runs on its own, which is worth the extra lines if you want to drop it into a notebook unchanged.

{% raw %}
```python
import math, random
from collections import Counter

# Same 61-document agri-support KB as the previous block, repeated so this
# block runs standalone. Skip to "fusion" below if you ran block 1.
RAW = """
maize fall armyworm scouting :: scout maize fields weekly for fall armyworm; check whorls for window pane damage and frass :: pest maize scouting
agri-4021 fertiliser blend :: top dress maize with AGRI-4021 blend at six weeks; apply 50 kg per acre near the stem :: fertiliser maize topdress
kephis seed certification :: buy certified seed carrying the KEPHIS label; unlabelled seed carries germination risk :: seed certification regulation
drip irrigation scheduling :: run drip lines early morning; umwagiliaji twice a week beats daily light watering :: irrigation water scheduling
soil ph testing :: test soil ph yearly; maize tolerates 5.5 to 7.0, add lime when below 5.5 :: soil ph amendment
post harvest storage :: dry grain to 13 percent moisture before storage; use hermetic bags against weevils :: storage postharvest moisture
aphid outbreak on beans :: aphids cluster under bean leaves; spray neem extract at dusk to protect pollinators :: pest beans spraying
kalro variety trial results :: KALRO trials show drought tolerant variety yields 2.1 tonnes per hectare under short rains :: variety drought yield
mbolea application timing :: apply mbolea at planting rather than after emergence so roots reach nutrients early :: fertiliser timing planting
wadudu monitoring traps :: hang yellow sticky traps at canopy height to count wadudu before spraying :: pest monitoring traps
coffee berry borer :: prune shade trees and strip fallen berries to break coffee berry borer cycle :: pest coffee pruning
rainfall onset forecast :: onset forecasts from the county met office guide planting dates; do not plant on a single wet day :: weather planting forecast
greenhouse ventilation :: open side vents by nine am; leaf curl signals heat stress inside the tunnel :: greenhouse heat ventilation
afa registration :: food exporters must register with AFA and keep traceability records per consignment :: regulation export traceability
irrigation water testing :: test irrigation water for salinity; above 2 ds per metre switch to tolerant crops :: water salinity testing
maize lethal necrosis :: rogue out plants with yellowing and dead heart; maize lethal necrosis spreads by thrips :: disease maize rogueing
tissue culture bananas :: tissue culture banana plantlets are disease free; harden them in shade for two weeks :: banana propagation seedling
compost heap layering :: layer dry stalks and green matter, turn every three weeks, keep heaps moist but not soaked :: soil compost organic
avocado export grade :: grade avocados by diameter; exporters reject fruit below 55 millimetres :: export avocado grading
poultry feed conversion :: track feed conversion weekly; a rise above 1.8 signals disease or poor ventilation :: poultry feed monitoring
agri-7733 foliar spray :: spray AGRI-7733 foliar at flowering only; later sprays leave residue above the export limit :: fertiliser spraying residue
storage pest fumigation :: fumigate silos under licensed supervision; phosphine needs seven days sealed :: storage pest fumigation
drought tolerant sorghum :: sorghum out yields maize in marginal rainfall; plant at onset on 75 centimetre rows :: variety drought sorghum
water pan construction :: line a farm water pan with clay and compact the floor to cut seepage losses :: water storage construction
leaf spot on kale :: remove lower kale leaves showing concentric spots and rotate beds for two seasons :: disease kale rotation
extension officer visits :: group requests for extension visits go through the ward agriculture office each quarter :: extension access services
mobile advisory sms :: subscribe to the county sms advisory; messages carry planting windows and pest alerts :: advisory sms access
soil erosion terraces :: cut terraces on slopes above eight percent to hold topsoil through the long rains :: soil erosion terracing
nursery shading :: shade vegetable nurseries at fifty percent until seedlings have four true leaves :: nursery shading seedling
milk cooling chain :: cool milk to four degrees within two hours; reject milk with added water :: dairy cooling quality
agri-1180 soil conditioner :: work AGRI-1180 conditioner into the top ten centimetres before planting potatoes :: soil amendment potato
fallow rotation legumes :: rotate cereals with legumes to rebuild nitrogen without extra fertiliser spend :: rotation legumes soil
irrigation pump maintenance :: clean pump filters weekly; a clogged filter cuts pressure and burns the motor :: irrigation maintenance equipment
market price information :: check daily wholesale prices at the county market board before harvesting for sale :: market price selling
seed rate calculation :: maize seed rate of ten kilograms per acre suits seventy five centimetre spacing :: seed spacing planting
livestock vaccination :: vaccinate cattle against east coast fever before the rains; keep the certificate :: livestock vaccination health
solar dryer fruit :: solar dry mango slices at fifty five degrees for two days to hold colour :: postharvest solar fruit
pest scouting records :: keep scouting records per plot; spray only when counts cross the action threshold :: pest records threshold
agri-5502 herbicide :: apply AGRI-5502 pre emergence within three days of planting; never on emerged maize :: herbicide weed timing
weed control mulching :: mulch with dry grass to suppress weeds and hold moisture through dry spells :: weed mulch moisture
bee keeping pollination :: place two hives per acre of avocado to lift fruit set during flowering :: pollination bees avocado
cold chain tomato :: pack tomatoes in crates not sacks; sacks crush fruit and cut market grade :: postharvest tomato packaging
salinity tolerant varieties :: where irrigation water is salty, plant salinity tolerant varieties and leach salts with rain :: water salinity variety
agronomy training days :: field days run every second thursday at the sub county showground, free to farmer groups :: extension training access
silage making :: chop forage to two centimetres, compact hard and cover to exclude air for silage :: livestock feed silage
mango fruit fly :: hang fruit fly traps and collect fallen fruit twice weekly to break the cycle :: pest mango traps
water harvesting gutter :: fit gutters to the iron roof and store runoff in a lined tank for the nursery :: water harvesting construction
agri-9081 foliar calcium :: AGRI-9081 foliar calcium prevents blossom end rot in tomatoes during fast growth :: fertiliser tomato calcium
soil carbon monitoring :: sample soil carbon every third season to track whether organic inputs are working :: soil carbon monitoring
price hedging cooperative :: cooperatives pool volumes to negotiate a forward price before harvest :: market cooperative selling
seed storage humidity :: store seed below forty percent humidity with a moisture indicator in the sack :: seed storage humidity
irrigation emitter clogging :: flush drip lines monthly; emitters clog faster with borehole water high in iron :: irrigation maintenance water
agri-6604 nematicide :: apply AGRI-6604 nematicide two weeks before tomato transplanting, not at planting :: pest nematicide tomato
livestock tick control :: spray cattle weekly in the wet season; ticks carry east coast fever :: livestock ticks spraying
post harvest loss audit :: weigh a sample after sorting to estimate post harvest loss per tonne stored :: postharvest loss records
agroforestry shade :: plant grevillea on field boundaries for timber without shading the maize rows :: agroforestry shade timber
fertiliser subsidy register :: the subsidised fertiliser register is verified at the ward office before collection :: fertiliser subsidy access
hive hygiene :: clean hives and replace old comb yearly to hold down varroa loads :: bees health maintenance
agri-2255 seed treatment :: treat seed with AGRI-2255 before planting to stop soil borne damping off :: seed treatment disease
farmer field school :: farmer field schools meet weekly through the season to compare plots side by side :: extension training access
soil test laboratory :: send composite samples to the accredited laboratory; results return in ten days :: soil testing access
"""

DOCS = []
for line in RAW.strip().splitlines():
    title, body, concepts = [p.strip() for p in line.split("::")]
    DOCS.append((title, body, set(concepts.split())))

QUERIES = [
    ("agri-4021", "literal", 1, set()), ("kephis", "literal", 2, set()),
    ("agri-7733", "literal", 20, set()), ("agri-1180", "literal", 30, set()),
    ("agri-5502", "literal", 38, set()), ("agri-9081", "literal", 47, set()),
    ("agri-6604", "literal", 52, set()), ("agri-2255", "literal", 58, set()),
    ("kalro", "literal", 7, set()), ("afa registration", "literal", 13, {"regulation"}),
    ("chemicals that stop grass growing", "paraphrase", 38, {"herbicide", "weed"}),
    ("keeping grain safe from insects after drying", "paraphrase", 5, {"storage", "postharvest"}),
    ("how do i know if my land is acidic", "paraphrase", 4, {"soil", "ph"}),
    ("giving plants food when they first go in the ground", "paraphrase", 8, {"fertiliser", "planting"}),
    ("a way to grow plants without rain", "paraphrase", 3, {"irrigation", "water"}),
    ("stopping soil washing down a hill", "paraphrase", 28, {"soil", "erosion"}),
    ("cooling milk quickly after milking", "paraphrase", 29, {"dairy", "cooling"}),
    ("insect that eats maize leaves in the funnel", "paraphrase", 0, {"pest", "maize"}),
    ("buying seed that is officially approved", "paraphrase", 2, {"seed", "certification"}),
    ("keeping tomatoes from being crushed on the way to market", "paraphrase", 42, {"postharvest", "tomato"}),
    ("agri-4021 when to apply", "mixed", 1, {"fertiliser"}),
    ("aphids neem", "mixed", 6, {"pest", "beans"}),
    ("drip lines iron", "mixed", 51, {"irrigation", "maintenance"}),
    ("sorghum rows marginal rain", "mixed", 22, {"variety", "drought"}),
    ("avocado hives", "mixed", 40, {"bees", "avocado"}),
    ("silage chop length", "mixed", 45, {"livestock", "feed"}),
    ("terraces slope percent", "mixed", 28, {"soil", "erosion"}),
    ("mango traps fallen fruit", "mixed", 46, {"pest", "mango"}),
    ("tomato crates sacks", "mixed", 42, {"postharvest", "tomato"}),
    ("kale concentric spots", "mixed", 24, {"disease", "kale"}),
]

def tokenize(text):
    out = []
    for tok in text.lower().replace("-", " ").split():
        tok = "".join(ch for ch in tok if ch.isalnum())
        if tok:
            out.append(tok)
    return out

class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.k1, self.b = k1, b
        self.tfs, self.lens, df = [], [], Counter()
        for title, body, _ in docs:
            tf = Counter(tokenize(title + " " + body))
            self.tfs.append(tf); self.lens.append(sum(tf.values())); df.update(tf.keys())
        self.n = len(docs)
        self.avgdl = sum(self.lens) / self.n
        self.idf = {w: math.log(1 + (self.n - d + 0.5) / (d + 0.5)) for w, d in df.items()}

    def scores(self, query):
        q, out = tokenize(query), []
        for i, tf in enumerate(self.tfs):
            norm = self.k1 * (1 - self.b + self.b * self.lens[i] / self.avgdl)
            s = 0.0
            for w in q:
                if w in tf:
                    f = tf[w]
                    s += self.idf[w] * (f * (self.k1 + 1)) / (f + norm)
            out.append(s)
        return out

class Dense:
    def __init__(self, docs, noise=0.12, seed=11):
        self.concepts = [d[2] for d in docs]
        self.noise, self.seed = noise, seed

    def scores(self, query_concepts):
        rng, out = random.Random(self.seed), []
        for cs in self.concepts:
            union = len(cs | query_concepts) or 1
            out.append(len(cs & query_concepts) / union + self.noise * rng.random())
        return out

# ---------------- fusion ----------------
def ranked(scores, n=10):
    return [i for i, _ in sorted(enumerate(scores), key=lambda p: (-p[1], p[0]))[:n]]

def rrf(score_lists, k=60, n=10):
    total = Counter()
    for scores in score_lists:
        for rank, doc in enumerate(ranked(scores, n), 1):
            total[doc] += 1.0 / (k + rank)
    return [d for d, _ in sorted(total.items(), key=lambda p: (-p[1], p[0]))[:n]]

def minmax(scores, n=10):
    top = ranked(scores, n)
    lo, hi = min(scores[d] for d in top), max(scores[d] for d in top)
    span = (hi - lo) or 1.0
    return {d: (scores[d] - lo) / span for d in top}

def weighted(lex, sem, alpha, n=10, normalise=True):
    nl, ns = minmax(lex, n), minmax(sem, n)
    total = {}
    for d in set(nl) | set(ns):
        a, b = (nl.get(d, 0.0), ns.get(d, 0.0)) if normalise else (lex[d], sem[d])
        total[d] = alpha * b + (1 - alpha) * a
    return [d for d, _ in sorted(total.items(), key=lambda p: (-p[1], p[0]))[:n]]

def rr(rl, gold):
    return next((1.0 / p for p, d in enumerate(rl, 1) if d == gold), 0.0)

def rec(rl, gold):
    return 1.0 if gold in rl else 0.0

bm25, dense = BM25(DOCS), Dense(DOCS)
LEX = [(q, c, g, bm25.scores(q), dense.scores(mean))
       for q, c, g, mean in QUERIES]

STRATEGIES = [
    ("bm25",         lambda l, s: ranked(l)),
    ("dense",        lambda l, s: ranked(s)),
    ("raw sum",      lambda l, s: weighted(l, s, 0.5, normalise=False)),
    ("min-max a=.5", lambda l, s: weighted(l, s, 0.5)),
    ("rrf k=0",      lambda l, s: rrf([l, s], 0)),
    ("rrf k=10",     lambda l, s: rrf([l, s], 10)),
    ("rrf k=60",     lambda l, s: rrf([l, s], 60)),
]

print("=== strategy comparison, 30 queries ===")
print(f"{'strategy':14s}{'R@1':>6s}{'R@10':>7s}{'MRR':>7s}{'lit':>8s}{'para':>8s}")
for name, fn in STRATEGIES:
    outs = [(r[1], fn(r[3], r[4]), r[2]) for r in LEX]
    r1 = sum(1.0 for c, o, g in outs if o[0] == g) / len(outs)
    r10 = sum(rec(o, g) for c, o, g in outs) / len(outs)
    mrr = sum(rr(o, g) for c, o, g in outs) / len(outs)
    lit = sum(rr(o, g) for c, o, g in outs if c == "literal") / sum(1 for c, _, _ in outs if c == "literal")
    par = sum(rr(o, g) for c, o, g in outs if c == "paraphrase") / sum(1 for c, _, _ in outs if c == "paraphrase")
    print(f"{name:14s}{r1:6.2f}{r10:7.2f}{mrr:7.3f}{lit:8.3f}{par:8.3f}")

print("\n=== tune alpha on paraphrase-heavy traffic, then serve literal queries ===")
dev = [r for r in LEX if r[1] in ("paraphrase", "mixed")]
best_a, best_dev = 0.5, -1.0
for i in range(11):
    a = i / 10
    m = sum(rr(weighted(r[3], r[4], a), r[2]) for r in dev) / len(dev)
    print(f"  alpha={a:.1f}  dev MRR={m:.3f}")
    if m > best_dev:
        best_a, best_dev = a, m
lit = [r for r in LEX if r[1] == "literal"]
tuned = sum(rr(weighted(r[3], r[4], best_a), r[2]) for r in lit) / len(lit)
half = sum(rr(weighted(r[3], r[4], 0.5), r[2]) for r in lit) / len(lit)
rrf60 = sum(rr(rrf([r[3], r[4]], 60), r[2]) for r in lit) / len(lit)
lex = sum(rr(ranked(r[3]), r[2]) for r in lit) / len(lit)
print(f"best alpha on dev: {best_a:.1f} (dev MRR {best_dev:.3f})")
print(f"literal MRR: tuned {tuned:.3f}  a=0.5 {half:.3f}  rrf60 {rrf60:.3f}  bm25 alone {lex:.3f}")

print("\n=== min-max normalisation window: top-10 vs top-50 ===")
moved = sum(1 for r in LEX
            if weighted(r[3], r[4], 0.5, n=10)[0] != weighted(r[3], r[4], 0.5, n=50)[0])
print(f"queries whose top-1 result changes: {moved} of {len(LEX)}")
```
{% endraw %}

```text
=== strategy comparison, 30 queries ===
strategy         R@1   R@10    MRR     lit    para
bm25            0.63   0.73  0.658   1.000   0.324
dense           0.40   0.57  0.459   0.077   0.800
raw sum         0.63   0.80  0.690   1.000   0.419
min-max a=.5    0.67   0.83  0.742   0.850   0.750
rrf k=0         0.67   0.83  0.744   0.850   0.750
rrf k=10        0.47   0.83  0.589   0.525   0.617
rrf k=60        0.43   0.83  0.572   0.525   0.567

=== tune alpha on paraphrase-heavy traffic, then serve literal queries ===
  alpha=0.0  dev MRR=0.492
  alpha=0.1  dev MRR=0.520
  alpha=0.2  dev MRR=0.525
  alpha=0.3  dev MRR=0.545
  alpha=0.4  dev MRR=0.554
  alpha=0.5  dev MRR=0.688
  alpha=0.6  dev MRR=0.682
  alpha=0.7  dev MRR=0.706
  alpha=0.8  dev MRR=0.706
  alpha=0.9  dev MRR=0.706
  alpha=1.0  dev MRR=0.650
best alpha on dev: 0.7 (dev MRR 0.706)
literal MRR: tuned 0.233  a=0.5 0.850  rrf60 0.525  bm25 alone 1.000

=== min-max normalisation window: top-10 vs top-50 ===
queries whose top-1 result changes: 6 of 30
```

Read the first table as three separate claims, because they land differently.

**Fusion doubles the recall gap.** Every fused strategy reaches 0.83 recall at ten against 0.73 and 0.57 for the single retrievers. That part is uncontroversial and it is the reason hybrid search exists: the candidate union contains answers neither list finds alone.

**Raw score addition barely participates.** At 0.690 MRR it beats BM25's 0.658 overall, which looks like a win until you read the class columns: literal stays at exactly 1.000, paraphrase moves 0.324 to 0.419. The semantic list is contributing tiebreaks, not rankings. If you ship this and check one aggregate number, you will conclude fusion works when what you shipped is BM25 with extra latency.

**`k` is not a constant you inherit.** `rrf k=0` scores 0.744 MRR, `k=10` gives 0.589, `k=60` gives 0.572. On this corpus, the setting both Elasticsearch and Azure default to is the worst of the three. The mechanism is the 1.15x ratio from the previous block: with two lists, flattening rank differences lets a noisy second list outvote a confident first list.

That direction of result is not an artifact of my toy corpus. Bruch, Gai and Ingber re-examined RRF against a tuned convex combination and found "NDCG swings wildly as a function of RRF parameters", with lower constants often beating the default: on MS MARCO, RRF at `(5, 5)` scored 0.435 NDCG@1000 against 0.425 for `(60, 60)`, and `(10, 4)` reached 0.451 ([arXiv:2210.11934](https://arxiv.org/abs/2210.11934), Table 3). Their headline result is that a two-parameter convex combination beats RRF in-domain and out-of-domain, and that RRF discards information by ignoring score distances.

Cormack's own pilot data shows the same flatness from the other side. MAP by `k` runs 0.2072 at `k=0`, 0.2145 at `k=60` and 0.2098 at `k=500`, a spread of under four percent across three orders of magnitude. The reason his `k=60` was safe is that he was fusing dozens of runs, where averaging many noisy lists is precisely the regime that wants heavy flattening. Fusing two lists is a different regime that inherited the same number.

Bruch's parametric reading of the formula gives you one more lever than the table above uses. Replace the single `k` with one constant per retriever, `1/(k_lex + rank_lex) + 1/(k_sem + rank_sem)`, and raising `k_lex` discounts the lexical list's contribution without touching its scores. That is how you say "this index is better on my traffic" while staying rank-only. The paper reports the direction flipping by dataset: on MS MARCO, an in-domain benchmark where the semantic model is strong, NDCG improved when `k_lex > k_sem`, and the opposite held on HotpotQA. Which means the two constants are also a fitted parameter, and inherit the same deployment risk as a fitted weight, just with a smaller blast radius because they cannot be skewed by score scales.

## Tuning a weight is not free

The second section of output is the part that changes how I would ship this.

Tuning on paraphrase-heavy traffic picks `alpha = 0.7`, the semantic-favouring end, at 0.706 dev MRR. Deployed against literal queries, that same weight scores 0.233. A fixed `alpha = 0.5` scores 0.850 on those queries, `k=60` RRF scores 0.525, and BM25 alone scores 1.000.

Two honest readings. The first is that a tuned weight is a statement about your query mix, not about retrieval in general, so it stops being valid the moment the mix shifts toward codes, names and jargon. The second is that the tuning grid itself was coarse: dev MRR jumps 0.554 to 0.688 between `alpha = 0.4` and `0.5`, then sits flat at 0.706 from 0.7 to 0.9. A finer grid would have landed somewhere in that plateau and produced a different deployment outcome.

This is a query-mix shift, not the dataset shift Bruch and colleagues study, and their finding runs the other way: a convex combination tuned on in-domain data held up better out-of-domain than a tuned RRF did. Both results are compatible. What they share is the warning: a weight is a fitted parameter, and parameters need their distribution checked before deployment. An unfitted `k = 60` has no such exposure, which is the real argument for RRF even when it scores lower on your benchmark.

## Normalisation windows are part of the contract

The last line: min-max normalisation over the top 10 versus the top 50 changes the top-1 answer for 6 of 30 queries. Nothing in the query changed, only which results were in the window when the scores were rescaled.

That is why OpenSearch makes normalisation an explicit, configurable processor rather than an implicit step, and why `min_max` and `l2` behave differently on the same pair of lists. It is also why Weaviate moved its hybrid default from rank-based fusion to `relativeScoreFusion` in 1.24, a score-based path its own engineers call "likely the better choice for most" ([Weaviate](https://weaviate.io/blog/hybrid-search-fusion-algorithms)).

The practical consequence: if you use score fusion, the window size is a hyperparameter. Write it down next to the weight, change it in the same commit as the weight, and evaluate both together.

## What to ship

| Situation | Merge | Why |
|---|---|---|
| Codes, IDs, names, jargon in queries | RRF with `k` between 0 and 10, or route literals to BM25 | Rank-only fusion cannot be fooled by score scales; low `k` keeps the winning list's confidence |
| Mostly natural-language questions, labelled queries available | Min-max convex combination, weight tuned on a held-out set | Reaches higher MRR than a fixed `k` when the mix is stable |
| No labels, unknown traffic | RRF with depth 50 or more, evaluate per query class | Costs no tuning and degrades gracefully |
| Two lists whose scores you control and can calibrate | Z-score or L2 normalisation before summing | Preserves score ratios that min-max destroys |

One implementation detail bites more often than the maths does. Elasticsearch requires `rank_window_size` to be at least the requested `size`, and if a child kNN retriever asks for more neighbours than the window allows, its results are truncated to the window. So the depth of your candidate lists is set by the fusion layer, not only by the vector query, and a semantic list that arrives short has fewer chances to contribute a document. Set the window first, then confirm each retriever can fill it. If you are writing the merge yourself, the same rule applies: take the top N from both lists with the same N, and drop anything outside it rather than inventing a rank for it.

Three habits that catch the failures above.

Print recall at ten per query class, not just overall. The 0.83 fused number hides that literal MRR fell from 1.000 to 0.850 under min-max, and a single aggregate will never show it.

Log both input ranks for every returned document. When a fusion result looks wrong, the ranks tell you whether the retriever missed the document or the merge demoted it, and those are different bugs with different fixes.

Sweep the constant on a query mix you can defend. If your production traffic is 40 percent identifiers, do not tune on a paraphrase-heavy dev set and then attribute the drop to the embedding model.

> **One warning about corpus size**
> The 0.83 ceiling here is a 61-document corpus. With a few hundred thousand documents the union of two top-50 lists covers far less of the answer space, fusion gains shrink, and recall at deeper cuts starts to matter more than MRR. Re-measure on your own corpus before copying the numbers.
{: .prompt-warning }

Thirty labelled queries is enough to show a direction and not enough to pin a constant. The dev-versus-literal split above is a deliberate mix change, not a significance test, and a two-point gap on thirty queries is noise by any standard. Before a constant goes into production, run the comparison on a few hundred labelled queries, pair it per query, and report the difference with a confidence interval; the [annotator agreement post](/posts/annotator-agreement-before-you-train/) covers why the labels themselves need checking first.

## Key takeaways

| Finding | Measured | What to do |
|---|---|---|
| The two retrievers fail on disjoint query classes | BM25 1.000 literal MRR, dense 0.077; dense 0.800 paraphrase, BM25 0.324 | Fuse, and report per class |
| Raw score addition reduces to the stronger retriever | paraphrase MRR 0.419 vs 0.750 for min-max | Normalise, or switch to ranks |
| `k=60` over-flattens a two-list fusion | rank1/rank10 falls from 10.00x at `k=0` to 1.15x at `k=60`; MRR 0.744 vs 0.572 | Tune `k`; default to 0-10 with two lists |
| A tuned weight is a property of the query mix | dev 0.706 becomes 0.233 on literal traffic | Pin the mix, or avoid fitted weights |
| The normalisation window is a hyperparameter | 6 of 30 top-1 answers change from window 10 to 50 | Version the window with the weight |

Fusion is a hedge against the query you cannot classify. Rank-based fusion is the cheaper hedge, because it cannot be fooled by the scale mismatch that breaks the arithmetic, at the price of throwing away score distances. The one thing worth measuring before shipping either is the same: per-class recall, with the window and the constant written down.

## References

- Cormack, Clarke and Büttcher, "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods", SIGIR 2009 — https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf
- Bruch, Gai and Ingber, "An Analysis of Fusion Functions for Hybrid Retrieval", arXiv:2210.11934 — https://arxiv.org/abs/2210.11934
- Elasticsearch reference, "Reciprocal rank fusion" (`rank_constant`, `rank_window_size`) — https://www.elastic.co/docs/reference/elasticsearch/rest-apis/reciprocal-rank-fusion
- Microsoft Learn, "Relevance scoring in hybrid search using Reciprocal Rank Fusion" — https://learn.microsoft.com/en-us/azure/search/hybrid-search-ranking
- OpenSearch documentation, "Normalization processor" (min_max, l2, z_score) — https://docs.opensearch.org/latest/search-plugins/search-pipelines/normalization-processor/
- Weaviate, "Unlocking the Power of Hybrid Search: Fusion Algorithms" — https://weaviate.io/blog/hybrid-search-fusion-algorithms
- Lucene API, `BM25Similarity` (k1 = 1.2, b = 0.75, idf formula) — https://lucene.apache.org/core/9_10_0/core/org/apache/lucene/search/similarities/BM25Similarity.html

## Related posts

- [The Denominator Is the Metric: auditing a retriever before you blame the model](/posts/rag-recall-at-k-denominator/) — how to compute recall so the number means something
- [Filtered Vector Search: sizing the candidate pool](/posts/filtered-vector-search-pool-sizing/) — why a filter shrinks the ANN pool and how to size it
- [Semantic Cache Thresholds: precision before you save a call](/posts/semantic-cache-threshold-precision/) — picking a similarity cutoff with data instead of instinct
- [Embedding Compression: what you lose when you shrink the index](/posts/embedding-compression-audit/) — measuring quality after quantisation
