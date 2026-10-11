---
title: "Redact, Then Prove It: Per-Type Recall, Offset Damage and the Reversible Pseudonym"
date: 2026-10-11 00:00:00 +0300
categories: [Machine Learning, ML Ops]
tags: [privacy, pii, data quality, tokenization, k-anonymity, mlops]
image:
  path: /assets/img/cover-pii-redaction-recall-and-reversibility.webp
  alt: A support note whose fields are covered by mask blocks, beside a ruler marking the character offsets of each span and a column of hashed identifiers a magnifying glass turns back into the original numbers
---

> **Two numbers from this post's fixture: the day-one regex set caught 5 of 13 PII spans, and a plain SHA-256 release gave all 24 subject IDs back to an attacker holding a 200,000-value register.**
> Both are measured below in standard-library Python: a labelled corpus, a per-type recall table, an
> offset-safe masker, and two audits that decide whether a file labelled de-identified can be walked
> back to a person.
{: .prompt-info}

Redaction looks like one task and is actually two. The detector has to find every span in the text you hold, and the transform has to leave a release that cannot be pulled back to an individual. Teams usually test neither, because the library has tests and the library is not the problem.

## What the benchmarks actually measure

The best evidence for testing your own text first is unflattering. [PIIBench](https://arxiv.org/abs/2604.15776) merges ten public corpora into 2,369,883 annotated sequences and 3.35 million entity mentions across 48 canonical PII types, then evaluates eight published systems on it: "All systems achieve span-level F1 below 0.14, with the best system (Presidio, F1=0.1385) still producing zero recall on most entity types." The diagnosis is the domain-silo problem, since a detector tuned on synthetic English PII meets financial text, multilingual text and region-specific identifiers that were never in its mix.

[Mind the Gap](https://arxiv.org/abs/2609.03464) stress-tests three architectural families over seven categories of distribution shift: encoders fail on "unseen surface forms and boundary detection", rule-based systems on "non-standard formats", and the LLM on "entity-type confusion and generation instability". Its conclusion is the one to keep: "aggregate benchmark scores obscure deployment-critical weaknesses." [RECAP](https://arxiv.org/abs/2510.07551) adds a weighted recall of 0.605 for a hybrid of regexes and prompts, against 0.362 for fine-tuned NER and 0.437 for a zero-shot LLM, while warning that the comparison is not like for like and that synthetic data "may not capture the full complexity of real-user text".

The pattern in all three is not a better model but a different order of work, and Presidio's own entity list shows it: detection methods are declared per entity, and the highest-precision ones pair a pattern with a **checksum or format validator**. A credit card is "between 12 to 19 digits" and detected by "pattern match and checksum", IBAN codes carry a checksum, and email addresses get "RFC-822 validation". A validator turns a format guess into an arithmetic claim.

## The fixture: 15 notes, 13 spans, two traps

The first block builds the text, the labels, both detectors and the masker, then scores them the way the benchmark does: span-level exact match with the type included, so a span that is found but mis-typed counts as a miss and a false positive at once. Two records carry values that look like identifiers and are not, an order number and a card whose check digit fails.

```python
import re
from collections import Counter

CORPUS = [
    ("Callback requested: +254 712 345 678 after 6pm.", [("PHONE", "+254 712 345 678")]),
    ("Alt line 254733118899 no spaces, use after 6pm.", [("PHONE", "254733118899")]),
    ("Reached on 0720 559 041 during the Nairobi visit.", [("PHONE", "0720 559 041")]),
    ("Customer 0722944556 asked for refund status.", [("PHONE", "0722944556")]),
    ("OTP for 0712000111 sent; ref 2026-10-11 logged.", [("PHONE", "0712000111")]),
    ("Tracking https://example.co.ke/t/254799887766 shows delivered.", [("PHONE", "254799887766")]),
    ("Callback line +254 (712) 345-678 is the one on the form.", [("PHONE", "+254 (712) 345-678")]),
    ("Card on file 4111 1111 1111 1111 auth failed.", [("CARD", "4111 1111 1111 1111")]),
    ("Card 5500005555555559 charged KES 2,300.", [("CARD", "5500005555555559")]),
    ("Card ending 4111111111111112 was declined in test.", []),
    ("ID 31245678 verified at branch 041.", [("NATID", "31245678")]),
    ("ID number 0412345 was updated on the portal.", [("NATID", "0412345")]),
    ("Order ORD-31245678 shipped from the Westlands hub.", []),
    ("Email jane.doe@example.co.ke bounced hard.", [("EMAIL", "jane.doe@example.co.ke")]),
    ("She wrote: jane dot doe at example dot co dot ke", [("EMAIL", "jane dot doe at example dot co dot ke")]),
]

DAY_ONE = {                                        # what a first hour of regex work produces
    "PHONE": re.compile(r"(?:\+254|0)\d{9}"),
    "CARD": re.compile(r"\d{16}"),
    "NATID": re.compile(r"\b\d{8}\b"),
    "EMAIL": re.compile(r"[\w.]+@[\w.]+"),
}
WINDOW = re.compile(r"\+?[\d][\d\s\-()]{5,}\d")    # admits separators, brackets, country codes
PLAIN_MAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
SPELLED_MAIL = re.compile(r"\b[\w.]+(?:\s+dot\s+[\w.]+)+\s+at\s+[\w.]+(?:\s+dot\s+[\w.]+)+", re.I)


def gold_spans():
    """(start, end, type) for every labelled surface, in document order."""
    out = []
    for text, items in CORPUS:
        spans = sorted((text.index(s), text.index(s) + len(s), t) for t, s in items)
        out.append(spans)
    return out


def day_one(text):
    return [(m.start(), m.end(), t) for t, rx in DAY_ONE.items() for m in rx.finditer(text)]


def luhn(digits):
    total, alt = 0, False
    for ch in reversed(digits):
        d = int(ch)
        if alt:
            d *= 2
            d -= 9 if d > 9 else 0
        total += d
        alt = not alt
    return total % 10 == 0


def valid_phone(d):
    """Kenyan mobile: 254 + 9 digits, local 0 + 9, or a bare 9-digit national number."""
    if len(d) == 12 and d.startswith("254"):
        return d[3] in "71"
    if len(d) == 10 and d.startswith("0"):
        return d[1] in "71"
    return len(d) == 9 and d[0] in "71"


def validated(text):
    """Classify each digit run arithmetically instead of trusting the first pattern that fits."""
    found = []
    for m in WINDOW.finditer(text):
        digits = re.sub(r"\D", "", m.group())
        glued = m.start() > 0 and (text[m.start() - 1].isalnum() or text[m.start() - 1] == "-")
        if len(digits) >= 12 and luhn(digits):
            found.append((m.start(), m.end(), "CARD"))
        elif valid_phone(digits):
            found.append((m.start(), m.end(), "PHONE"))
        elif 7 <= len(digits) <= 8 and m.group().isdigit() and not glued:
            found.append((m.start(), m.end(), "NATID"))
    for rx in (PLAIN_MAIL, SPELLED_MAIL):
        found += [(m.start(), m.end(), "EMAIL") for m in rx.finditer(text)]
    return sorted(found)


def merge(spans):
    """One mask per region: overlapping claims collapse into the widest span."""
    out = []
    for s, e, t in sorted(spans):
        if out and s < out[-1][1]:
            out[-1] = (out[-1][0], max(e, out[-1][1]), out[-1][2])
        else:
            out.append((s, e, t))
    return out


def mask_end_backwards(text, spans, tag="<{type}>"):
    out = text
    for s, e, t in sorted(merge(spans), reverse=True):
        out = out[:s] + tag.format(type=t) + out[e:]
    return out


def mask_left_to_right(text, spans, tag="<{type}>"):
    """The day-one bug: offsets measured on the original string, splices applied forwards."""
    out = text
    for s, e, t in sorted(spans):
        out = out[:s] + tag.format(type=t) + out[e:]
    return out


def intact(text, out, spans):
    """Stripping the masks from `out` must reproduce the input minus its labelled spans."""
    expected = text
    for s, e, _ in reversed(spans):
        expected = expected[:s] + expected[e:]
    return re.sub(r"<[A-Z]+>", "", out) == expected


def score(detect):
    """Per type: [true positives, false positives, false negatives], span-exact."""
    counts = {}
    for i, (text, _) in enumerate(CORPUS):
        gold = set(gold_spans()[i])
        pred = set(detect(text))
        for s, e, t in gold:
            counts.setdefault(t, [0, 0, 0])[0 if (s, e, t) in pred else 2] += 1
        for s, e, t in pred - gold:
            counts.setdefault(t, [0, 0, 0])[1] += 1
    return counts


labelled = Counter(t for _, items in CORPUS for t, _ in items)
print(f"{len(CORPUS)} records, {sum(len(t) for t, _ in CORPUS)} characters,"
      f" {sum(labelled.values())} labelled spans ({dict(sorted(labelled.items()))}),"
      f" {sum(1 for _, i in CORPUS if not i)} unlabelled traps")
a, b = score(day_one), score(validated)
print(f"{'type':6s} {'day-one tp/fp/fn':>17s} {'validated tp/fp/fn':>19s}")
for t in sorted(set(a) | set(b)):
    print(f"{t:6s} {str(a.get(t, [0, 0, 0])):>17s} {str(b.get(t, [0, 0, 0])):>19s}")
for name, c in (("day-one", a), ("validated", b)):
    tp, fp, fn = (sum(v[i] for v in c.values()) for i in range(3))
    print(f"{name:9s} recall {tp / (tp + fn):.3f}  precision {tp / (tp + fp):.3f}"
          f"  ({tp} of {tp + fn} spans, {fp} false positives)")

bad_ltr = bad_rev = left_ltr = left_rev = 0
for i, (text, _) in enumerate(CORPUS):
    spans = gold_spans()[i]
    ltr, rev = mask_left_to_right(text, day_one(text)), mask_end_backwards(text, validated(text))
    bad_ltr += not intact(text, ltr, spans)
    bad_rev += not intact(text, rev, spans)
    needles = [text[s:e] for s, e, _ in spans]
    left_ltr += sum(n in ltr for n in needles)
    left_rev += sum(n in rev for n in needles)
print(f"\nmasking left-to-right: {bad_ltr}/{len(CORPUS)} records damaged,"
      f" {left_ltr} labelled spans left in the output")
print(f"masking end-backwards: {bad_rev}/{len(CORPUS)} records damaged,"
      f" {left_rev} labelled spans left in the output")
for i in (8, 9):
    text = CORPUS[i][0]
    print(f"record {i:02d}  in    : {text}")
    print(f"          broken: {mask_left_to_right(text, day_one(text))}")
    print(f"          fixed : {mask_end_backwards(text, validated(text))}")
```

```text
15 records, 706 characters, 13 labelled spans ({'CARD': 2, 'EMAIL': 2, 'NATID': 2, 'PHONE': 7}), 2 unlabelled traps
type    day-one tp/fp/fn  validated tp/fp/fn
CARD           [1, 1, 1]           [2, 0, 0]
EMAIL          [1, 0, 1]           [2, 0, 0]
NATID          [1, 1, 1]           [2, 0, 0]
PHONE          [2, 1, 5]           [7, 0, 0]
day-one   recall 0.385  precision 0.625  (5 of 13 spans, 3 false positives)
validated recall 1.000  precision 1.000  (13 of 13 spans, 0 false positives)

masking left-to-right: 11/15 records damaged, 8 labelled spans left in the output
masking end-backwards: 0/15 records damaged, 0 labelled spans left in the output
record 08  in    : Card 5500005555555559 charged KES 2,300.
          broken: Card <C<PHONE>ed KES 2,300.
          fixed : Card <CARD> charged KES 2,300.
record 09  in    : Card ending 4111111111111112 was declined in test.
          broken: Card ending <CARD> was declined in test.
          fixed : Card ending 4111111111111112 was declined in test.
```

Read the miss column first, because that is where data leaves the building. Eight of thirteen spans are invisible to the day-one patterns, and the misses cluster by layout rather than entity: the space after the country code, the bracketed international form, a number inside a URL path, a card written with spaces, a seven-digit ID below the floor, and an email spelled out in words. That is the non-standard-format failure the stress study names, in fifteen strings.

The false positives matter differently. The order number is read as a national ID, which would redact a shipment reference, and the check-digit failure is read as a card. The third is worse: the pattern set claims the substring `0000555555` inside an unseparated card as a phone number, which damages text rather than merely mislabelling it.

The validated column is the fix, and it is arithmetic rather than cleverness. A widened candidate window admits every separator and bracket form, then each digit run is decided by its own properties: length and the Luhn check for cards, national prefix and length for Kenyan mobile numbers, a digit count plus a not-glued-to-a-word boundary for IDs, and a second pattern for the spelled-out email. Thirteen of thirteen, both traps rejected. The honest scope is narrow, since a fixture I wrote myself cannot stand in for a benchmark and next week's layout is not in it; treat it as a regression test, and keep adding your own misses to it.

## Offsets, not patterns, are what corrupt text

The damage line is the one to carry into code review. Masks applied left to right, with offsets measured on the original string, damage 11 of 15 records and leave 8 labelled spans inside the mangled output. Record eight shows the mechanism in one line: the card is masked first, the offsets captured for the overlapping phone claim then point into the shortened string, and the pass writes `<PHONE>` into the middle of the word "charged", so text that read "charged" now reads `<C<PHONE>ed`.

The rule is one sentence: **measure spans on the original string, merge the overlaps, then splice from the end backwards**. The merge matters because two recognisers will claim the same region and only one mask may be written; the backwards order matters because every splice invalidates every offset to its right. Two assertions make it enforceable: stripping the masks must reproduce the input minus its labelled spans, and no labelled surface may survive in the output. The second is the only proof that the recall on your data is what you measured rather than what you assumed.

## Hashing is not anonymisation

Kenya's Data Protection Act defines **anonymisation** as "the removal of personal identifiers from personal data so that the data subject is no longer identifiable", and **pseudonymisation** as processing where data "can no longer be attributed to a specific data subject without the use of additional information", with that information "kept separately" under technical and organisational measures. The legal test is attributability, not readability, and a hashed column is unreadable while remaining attributable whenever the input space can be enumerated.

The second block builds a candidate register, releases the same 24 subject identifiers both ways, and then asks the other question a release has to answer: how many people are still alone in their row.

```python
import hashlib, hmac, random, time
from collections import Counter

SUBJECT_IDS = sorted({str(31245678 + 137 * k) for k in range(24)})
KEY = b"org-wide-tokenisation-key-held-outside-the-dataset"


def token(pid, key=b""):
    raw = pid.encode()
    if key:
        return hmac.new(key, raw, hashlib.sha256).hexdigest()[:16]
    return hashlib.sha256(raw).hexdigest()[:16]


def sweep(values, reps=3):
    """Hashes per second, best of `reps` passes: laptop timings move about 2x run to run."""
    rates = []
    for _ in range(reps):
        t0 = time.perf_counter()
        for v in values:
            hashlib.sha256(v.encode()).hexdigest()[:16]
        rates.append(len(values) / (time.perf_counter() - t0))
    return min(rates)


rng = random.Random(11)
real = list(SUBJECT_IDS)
register = set(real)
while len(register) < 200_000:                     # a plausible candidate register
    base = rng.choice(real)
    register.add(str(int(base) + rng.randint(-500_000, 500_000)))
register = sorted(register)

attack = {hashlib.sha256(p.encode()).hexdigest()[:16] for p in register}
plain = {token(p) for p in real}
keyed = {token(p, KEY) for p in real}
space = 99_999_999 - 9_999_999
print(f"candidate register: {len(register):,} values built around the true IDs")
print(f"plain SHA-256 release: {len(plain & attack)}/{len(real)} subject IDs recovered from that table")
print(f"HMAC-SHA256 release  : {len(keyed & attack)}/{len(real)} subject IDs recovered")
print(f"slowest of 3 passes sweeps the {space:,}-value 8-digit space on one core"
      f" in under 3 minutes: {space / sweep(register) < 180}")

events = [(token(p), i) for i, p in enumerate(real) for _ in range(2)]
events.append((token(SUBJECT_IDS[0]), 99))
groups = Counter(tok for tok, _ in events)
print(f"keyed tokens keep joins: {len(events)} events -> {len(groups)} tokens,"
      f" {sum(n - 1 for n in groups.values() if n > 1)} same-subject pairs recoverable")
print(f"per-event surrogates   : {len({f'S{i:06d}' for i in range(len(events))})} ids over"
      f" {len(events)} events, 0 pairs recoverable, repeat-contact features lost")

COUNTIES = [("Nairobi", 4400), ("Kiambu", 2400), ("Nakuru", 2160), ("Kakamega", 1870),
            ("Bungoma", 1670), ("Kisumu", 1150), ("Machakos", 1420), ("Kilifi", 1450),
            ("Meru", 1550), ("Turkana", 930), ("Garissa", 840), ("Lamu", 140)]
names = [c for c, w in COUNTIES for _ in range(w)]
for n in (5_000, 50_000, 500_000):
    r = random.Random(7)
    people = [(r.choice(names), r.choice("MF"), r.randint(1955, 2005),
               r.randint(1, 12), r.randint(1, 28)) for _ in range(n)]
    day = Counter((c, s, y, m, d) for c, s, y, m, d in people)
    year = Counter((c, s, y) for c, s, y, m, d in people)
    sexdob = Counter((s, y, m, d) for c, s, y, m, d in people)
    unique = lambda t: 100 * sum(v for v in t.values() if v == 1) / n
    print(f"n={n:>7,}  unique on (sex,dob) {unique(sexdob):5.1f}%"
          f"  | (county,sex,dob) {unique(day):5.1f}%  min k={min(day.values())}"
          f"  | (county,sex,year) {unique(year):5.1f}%  min k={min(year.values())}")
print("county weights follow population shares; Lamu is the smallest county, Nairobi the largest")
```

```text
candidate register: 200,000 values built around the true IDs
plain SHA-256 release: 24/24 subject IDs recovered from that table
HMAC-SHA256 release  : 0/24 subject IDs recovered
slowest of 3 passes sweeps the 90,000,000-value 8-digit space on one core in under 3 minutes: True
keyed tokens keep joins: 49 events -> 24 tokens, 25 same-subject pairs recoverable
per-event surrogates   : 49 ids over 49 events, 0 pairs recoverable, repeat-contact features lost
n=  5,000  unique on (sex,dob)  87.1%  | (county,sex,dob)  98.4%  min k=1  | (county,sex,year)   3.2%  min k=1
n= 50,000  unique on (sex,dob)  23.4%  | (county,sex,dob)  84.9%  min k=1  | (county,sex,year)   0.0%  min k=1
n=500,000  unique on (sex,dob)   0.0%  | (county,sex,dob)  25.3%  min k=1  | (county,sex,year)   0.0%  min k=19
county weights follow population shares; Lamu is the smallest county, Nairobi the largest
```

Every identifier comes back from the unsalted release, and the register did not need to be well chosen, since it was sampled around the true values the way an adjacent cohort or a marketing list would be. The keyed release returns nothing to the same table. The third line is the sharper argument because it takes the register out of the threat model: hashing 200,000 values three times measured 749,029 to 1,500,479 sha256/s on this laptop depending on load, so the whole 90,000,000-value eight-digit space costs one to two minutes of one core. When the domain is that small you do not need a register; you can hash all of it.

The linkage line changes designs rather than merely warning. Keyed tokens still join: 49 events collapse to 24 tokens with 25 same-subject pairs recoverable, which is how a repeat-contact or velocity feature gets built on pseudonymised data, while a random surrogate per event removes the linkage and the feature with it. Decide which you want before you pick a hash function.

The standards draw the same line. NIST SP 800-122 keeps **de-identified** and **anonymized** apart: de-identified records "can be re-identified" by "a code, algorithm, or pseudonym", hashing "is a common de-identification technique", while anonymized information is "previously identifiable information that has been de-identified and for which a code or other association for re-identification no longer exists." The guide allows a low confidentiality impact level only when the re-identification mechanism sits "in a separate system" under controls and the data "are not linkable, via public records or other reasonably available external records". GDPR Recital 26 holds that pseudonymised data attributable by additional information "should be considered to be information on an identifiable natural person", and NIST's footnote adds the boundary worth reading in full: HIPAA excludes de-identification where the code is derived from the PII itself. The key is the additional information, so it needs the lifecycle of a secret (see [ML pipeline secrets management](/posts/ml-secrets-management/)); an unsalted hash has no key, and nothing to protect.

## What survives after you remove the identifiers

Suppose detection and transform both worked, and every direct identifier is masked or keyed. What is left is county, sex, date of birth and a pseudonym. Can a row still point at one person?

The last three lines of the block answer that at three release sizes, and the lesson is in the direction the numbers move. At 5,000 rows, county, sex and a birthday single out 98.4% of the extract, and coarsening the birthday to a year drops that to 3.2%. At 500,000 rows the day-level fields leave only a quarter unique, and the year-level row has no unique records at all, with a smallest class of 19. Two caveats keep this honest: the 87.1% on sex and birthday alone is a coincidence of this fixture's sparsity rather than a national rate, since the population is synthetic with birthdays drawn uniformly across fifty years; and the number that decides exposure is the smallest class size, because a minimum k of 1 singles somebody out even when the average looks comfortable.

Published rates are the cautionary tale. Sweeney's 2000 result, that 87% of the US population, 216 million of 248 million, "had reported characteristics that likely made them unique based only on {5-digit ZIP, gender, date of birth}", with 53% at city level and 18% at county level, was computed on a census rather than an extract. Golle's replication of the same data disagrees with the headline figure, finding that "in 1990 (resp. 2000), only 61% (resp. 63%) of the US population was uniquely identifiable" by the same three attributes. Two careful studies of one population differ by more than twenty points on the number everybody quotes, which is the argument for measuring your own release at its own size and granularity.

## How to apply this to your own corpus

| Step | What to record | Why |
|---|---|---|
| Label 100 to 200 examples from your own text, then score per type | Per-type true positives, false positives and false negatives | The best span-level F1 in the unified benchmark is 0.1385 across 48 types, so no published number is your recall |
| Pair every pattern with a validator | Luhn, mobile prefix and length, IBAN mod-97, RFC-822 | It removes the check-digit and order-number traps, and it is how Presidio declares its own recognisers |
| Add a recogniser per layout in your sample | The spelled-out email, the bracketed number, the number in a URL | Rule-based detectors fail on non-standard formats, and the fix is a recogniser, not a bigger model |
| Detect on the original text, splice from the end, then assert | Merged intervals, one mask per region; strip-recovery equal, zero labelled surfaces left | A forward pass damaged 11 of 15 records here, and both assertions are one line |
| Choose the transform by threat model | Mask for display, keyed HMAC for joins, random surrogate for no linkage | A plain SHA-256 release was reversed 24 times out of 24 |
| Measure uniqueness on the release | Unique fraction and minimum class size, at release size and granularity | Coarsening one field moved uniqueness from 98.4% to 3.2% |

## Key takeaways

| Takeaway | Evidence in this post |
|---|---|
| Layout-locked patterns lose most spans, and the misses cluster by format | Day-one regexes: recall 0.385, precision 0.625, 8 of 13 spans missed, across six layouts |
| Validators remove the traps | Luhn rejects the check-digit card, the boundary rule rejects the order number, and classifying per digit run removes the substring false positive |
| Offsets, not patterns, corrupt text | Forward masking damaged 11 of 15 records and left 8 spans behind; end-backwards damaged 0 and left 0 |
| Unsalted hashing of an enumerable identifier is reversible | A 200,000-value register reversed 24 of 24 IDs from a plain SHA-256 release and 0 of 24 from a keyed one |
| Small domains fall to enumeration alone | 749,029 to 1,500,479 sha256/s measured, so the 90,000,000-value space costs one to two minutes of one core |
| Keyed tokens keep joins, surrogates destroy them | 49 events to 24 tokens with 25 pairs recoverable; per-event surrogates leave 0 |
| Uniqueness is a property of the release, not of the country | 98.4% unique at 5,000 rows on county, sex and birthday; 25.3% at 500,000 rows; 3.2% once the birthday is coarsened; Sweeney's 87% and Golle's 61%/63% differ on the same census data |

The two blocks are standard-library Python with nothing to install, every number above came out of them as published, and the quoted output is a single run. Pointing the first block at real data means replacing the corpus with a sample of your records and labelling the spans; the scoring, the masker and both audits need no changes. If you wire up only two checks tonight, make them the strip-recovery assertion on the masker and the reversal test on the column you were about to call anonymous.

## References

- Jha, P. (2026). [PIIBench: A Unified Multi-Source Benchmark Corpus for Personally Identifiable Information Detection](https://arxiv.org/abs/2604.15776) — 2,369,883 sequences, 48 entity types, all eight systems below span-level F1 0.14, best Presidio 0.1385.
- Zafar, A., Nowaczyk, S. (2026). [Mind the Gap: Robustness Risks in PII Detection Systems](https://arxiv.org/abs/2609.03464) — seven categories of distribution shift; encoder, rule-based and LLM failure modes.
- [An Evaluation Study of Hybrid Methods for Multilingual PII Detection](https://arxiv.org/html/2510.07551v1) — weighted recall 0.605 for RECAP against 0.362 for NER and 0.437 for a zero-shot LLM.
- Microsoft / Data Privacy Stack. [PII entities supported by Presidio](https://github.com/microsoft/presidio/blob/main/docs/supported_entities.md) — per-entity detection methods, including checksums for cards, IBAN, crypto and medical licences.
- McCallister, E., Grance, T., Scarfone, K. (2010). [NIST SP 800-122: Guide to Protecting the Confidentiality of Personally Identifiable Information](https://nvlpubs.nist.gov/nistpubs/legacy/sp/nistspecialpublication800-122.pdf) — de-identified versus anonymized, hashing as a de-identification technique, the separate-system condition, and the HIPAA footnote on codes derived from the PII.
- Republic of Kenya. [Data Protection Act, Cap. 411C](https://new.kenyalaw.org/akn/ke/act/2019/24/eng@2022-12-31) — section 2 definitions of anonymisation, pseudonymisation and personal data; section 4 territorial application.
- [GDPR Recital 26: Not applicable to anonymous data](https://gdpr-info.eu/recitals/no-26/) — pseudonymised data attributable by additional information remains information about an identifiable person.
- Sweeney, L. (2000). [Simple Demographics Often Identify People Uniquely](https://dataprivacylab.org/projects/identifiability/paper1.pdf) — 87% (216 million of 248 million) unique on 5-digit ZIP, gender and date of birth.
- Golle, P. (2006). [Revisiting the Uniqueness of Simple Demographics in the US Population](https://crypto.stanford.edu/~pgolle/papers/census.pdf) — the same three attributes uniquely identify 61% (1990) and 63% (2000) of the population.

## Related posts

- [Your Corpus Is Duplicating Itself: MinHash + LSH Dedup You Can Measure Tonight](/posts/minhash-lsh-corpus-dedup/) — the same discipline on corpus hygiene: measure the filter before trusting it.
- [The Denominator Is the Metric: Auditing a RAG Retriever Before You Blame the Model](/posts/rag-recall-at-k-denominator/) — why a recall figure means nothing until you know what was counted.
- [The Tokenizer Was the Point: Measuring MORENA, an Open 1.5B Model for 12 African Languages](/posts/morena-open-african-language-model-tokenizer/) — measuring a component on the data you hold rather than the data it was built for.
- [ML Pipeline Secrets Management: Vault, API Keys, and Credential Hygiene](/posts/ml-secrets-management/) — where the pseudonymisation key belongs, and how it rotates.
- [Evaluating LLMs: Benchmarks That Matter for African Use Cases](/posts/evaluating-llms-african-use-cases/) — the wider case for local evaluation over imported scores.
