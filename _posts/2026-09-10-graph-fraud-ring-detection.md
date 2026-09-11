---
title: "Fraud Rings Are a Graph Problem: Detecting Mule Networks in Transaction Graphs"
date: 2026-09-10 00:00:00 +0300
categories: [Machine Learning, Fintech]
tags: [graph ml, fraud detection, community detection, money mules, graph neural networks, transaction networks]
image:
  path: /assets/img/cover-graph-fraud-ring-detection.webp
  alt: A transaction network with a dense cluster of accounts linked by shared devices and phones highlighted as a ring, closed money loops arcing back on themselves, and a single weak shared-device link greyed out as a false positive
---

## The fraud is a network, not an account

In June, October and November 2023, police in 26 countries arrested **1,013 people** and identified **10,759 money mules** — account-holders who lend their names to move stolen money. The action exposed **over €100 million** in losses and prevented another €32 million ([OCCRP on Europol's EMMA 9](https://www.occrp.org/en/news/global-operation-targets-money-mules)). An eight-month sweep from November 2025 to June 2026, **Operation Jackal IV**, ended in **58 arrests and 263 identified suspects across 22 countries**; in South Africa it blocked **257 bank accounts** and seized $2.67 million ([BBC](https://www.bbc.com/news/articles/cq5xdnxppl4o), [Sahara Reporters](https://saharareporters.com/2026/08/25/operation-jackal-interpol-arrests-58-persons-identifies-over-260-west-african-organised)).

Read those numbers as a data structure and they say one thing: the unit of modern fraud is not the fraudulent account. It is the **ring** — a handful of accounts, devices and phone numbers wired together to move money fast enough that no single account looks wrong.

> **What this post adds**
> Our [mobile-money features post](/posts/fraud-ml-mobile-money/) asked which *features* catch novel fraud, our [Sidian mule anatomy](/posts/sidian-bank-mule-heist/) traced *one* dispersal, and our [drift post](/posts/fraud-model-drift-monitoring/) asked when a model goes stale. This post is the layer underneath: the **transaction graph**, and how to find a ring in it before the money leaves.
{: .prompt-info }

## Why per-account scoring cannot see a ring

A classic fraud model scores one transaction, or one account, at a time. That works for carding and lone attackers, but a ring is a *relational* property: each member's behaviour is unremarkable on its own, and the signal lives entirely in the connections between them.

Neo4j's fraud reference architecture puts it plainly: "The individual transaction often looks legitimate. The fraud signal only becomes apparent when analyzing the network of relationships, shared devices, common IP addresses, or circular money flows" ([Neo4j](https://neo4j.com/developer/industry-use-cases/finserv/retail-banking/ieee-cis-fraud-graphs/)). A 2024 review of graph neural networks for financial fraud, surveying more than 100 studies, agrees they outperform traditional methods at capturing relational patterns ([Cheng et al., arXiv:2411.05815](https://arxiv.org/abs/2411.05815)).

Cifas, the UK's fraud-prevention service, recorded **more than 444,000 cases** on its National Fraud Database in 2025 — over **1,200 a day** — with organised crime groups "operating across borders and targeting multiple sectors simultaneously" ([Cifas Fraudscape 2026](https://www.cifas.org.uk/newsroom/fraudscape2026)). The target is a group; the model must reason about groups.

## Four graph signals that expose a ring

A transaction graph is simple to state: **accounts are nodes; payments, shared devices, phone numbers and identity documents are edges.** Rings are found not by a cleverer per-account score but by four structural patterns.

| Signal | Graph operation | What it catches |
|---|---|---|
| Shared identity | **Hard-link components** — union accounts sharing a phone, card or national ID | Accounts controlled by one person |
| Shared behaviour | **Soft-link clusters** — accounts on one device, IP or cookie | Fraud farms, shared devices |
| Dense subgraph | **Community detection** (Louvain/Leiden) | Rings as "fraud islands" distinct from the legitimate giant component |
| Circular flow | **Cycle detection** over money edges | Layering — money that returns to where it started |

The strongest evidence for the first two is a December 2025 paper on payment-platform clustering. It separates **hard links** — high-confidence identity relationships like phone numbers, credit cards and national IDs — from **soft links** — behavioural associations like device fingerprints, cookies and IPs. On a real payment platform, collapsing hard-link **connected components** into super-nodes, then clustering the soft-link graph, cut the graph from **25 million to 7.7 million nodes** while **doubling detection coverage** ([arXiv:2512.19061](https://arxiv.org/abs/2512.19061)).

That two-tier idea is the method in miniature: hard links tell you *these accounts are the same actor* (cheap, high-precision); soft links tell you *this looks like an operation* (higher coverage — and where false positives breed).

## A worked ring detection, in one script

The method fits in a few dozen lines of standard-library Python. The batch below carries a cash-out ring (four accounts on one device *and* one phone), a layering ring (a three-hop loop on one phone), ordinary one-hop payments, and a trap — a household sharing a single device.

{% raw %}
```python
# Graph-based fraud-ring detection — illustrative transaction network.
# Nodes are accounts; edges are payments plus shared-entity links (device, phone).
from collections import defaultdict

# (payer, payee, amount_ksh, device, phone) — a small mixed batch of real-looking traffic
payments = [
    # Ring A — cash-out fan-out: one victim, four accounts, ONE device + ONE phone
    ("victim_7712", "acct_A1", 471300, "dev_A1",    "254-700-0001"),
    ("acct_A1",     "acct_A2", 300000, "dev_RINGA", "254-111-1111"),
    ("acct_A1",     "acct_A3", 251300, "dev_RINGA", "254-111-1111"),
    ("acct_A2",     "acct_A4", 120000, "dev_RINGA", "254-111-1111"),
    ("acct_A3",     "acct_A4",  90000, "dev_RINGA", "254-111-1111"),
    ("acct_A4",     "cashout_9", 180000, "dev_RINGA", "254-111-1111"),
    # Ring B — circular layering: B1 -> B2 -> B3 -> B1, one phone, no shared device
    ("acct_B1", "acct_B2", 200000, "dev_B1", "254-222-2222"),
    ("acct_B2", "acct_B3", 195000, "dev_B2", "254-222-2222"),
    ("acct_B3", "acct_B1", 190000, "dev_B3", "254-222-2222"),
    # Legitimate traffic (unlinked, single-hop)
    ("acct_L1", "acct_L2", 45000, "dev_L1", "254-300-0001"),
    ("acct_L3", "acct_L4", 12000, "dev_L3", "254-300-0002"),
    ("acct_L5", "acct_L6", 88000, "dev_L5", "254-300-0003"),
    # False-positive trap: a household SHARES one device but nothing else
    ("acct_F1", "acct_F2", 8000, "dev_FAM", "254-400-0001"),
    ("acct_F3", "acct_F2", 6000, "dev_FAM", "254-400-0002"),
]

nodes, out_edges, in_edges = set(), defaultdict(set), defaultdict(set)
by_device, by_phone, amount = defaultdict(set), defaultdict(set), {}
for payer, payee, amt, dev, phone in payments:
    nodes.update((payer, payee))
    out_edges[payer].add(payee)
    in_edges[payee].add(payer)
    by_device[dev].add(payer)
    by_phone[phone].add(payer)
    amount[(payer, payee)] = amt

def components(link_index):
    """Union-find over accounts that share the same entity (phone or device)."""
    parent = {n: n for n in nodes}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for members in link_index.values():
        members = sorted(members)
        for other in members[1:]:
            parent[find(other)] = find(members[0])
    groups = defaultdict(set)
    for n in nodes:
        groups[find(n)].add(n)
    return {frozenset(g) for g in groups.values() if len(g) > 1}

def cycles(max_len=3):
    """Simple directed cycles up to max_len hops — the layering signature."""
    found = set()
    def walk(start, node, path):
        if len(path) > max_len:
            return
        for nxt in out_edges[node]:
            if nxt == start and len(path) >= 2:
                found.add(tuple(sorted(path)))
            elif nxt not in path:
                walk(start, nxt, path + [nxt])
    for s in nodes:
        walk(s, s, [s])
    return found

hard_comp = components(by_phone)          # high-confidence identity links
soft_comp = components(by_device)         # behavioural / device links
loops = cycles()

# Ring score: each signal is weak alone; a ring earns points by CORROBORATION
score = defaultdict(int)
for comp in soft_comp:                    # shared device    -> +2
    for acct in comp:
        score[acct] += 2
for comp in hard_comp:                    # shared phone/ID  -> +1
    for acct in comp:
        score[acct] += 1
for acct in nodes:
    if in_edges[acct] and out_edges[acct]:
        score[acct] += 1                  # passes funds through
    if len(out_edges[acct]) >= 2:
        score[acct] += 1                  # fans out to >=2 destinations
for loop in loops:                        # closed money loop -> +1
    for acct in loop:
        score[acct] += 1

print("=== Shared-device clusters (soft link) ===")
for dev, members in sorted(by_device.items()):
    if len(members) > 1:
        print(f"  {dev:9s} -> {', '.join(sorted(members))}   [{len(members)} accounts]")

print("\n=== Shared-phone components (hard identity link) ===")
for comp in sorted(hard_comp, key=lambda c: sorted(c)[0]):
    print(f"  {' + '.join(sorted(comp))}")

print("\n=== Money-flow cycles (layering loops, <=3 hops) ===")
for loop in sorted(loops):
    legs = [amount.get((loop[i], loop[(i + 1) % len(loop)]), 0) for i in range(len(loop))]
    print(f"  {' -> '.join(loop)} -> {loop[0]}   (KSh {min(legs):,}-{max(legs):,} per leg)")

print("\n=== Ring scores (device + identity + loop + flow shape) ===")
print(f"  {'account':<12}{'device':>8}{'phone':>7}{'loop':>6}{'thru':>6}{'fan':>5}{'score':>7}  verdict")
for acct, sc in sorted(score.items(), key=lambda kv: (-kv[1], kv[0])):
    if sc == 0:
        continue
    dev = any(acct in m for m in soft_comp)
    ph = any(acct in c for c in hard_comp)
    lp = any(acct in l for l in loops)
    thru = bool(in_edges[acct] and out_edges[acct])
    fan = len(out_edges[acct]) >= 2
    verdict = "FLAG" if sc >= 3 else "REVIEW" if sc == 2 else "-"
    print(f"  {acct:<12}{'+' if dev else '-':>8}{'+' if ph else '-':>7}{'+' if lp else '-':>6}"
          f"{'+' if thru else '-':>6}{'+' if fan else '-':>5}{sc:>7}  {verdict}")

flagged = sorted(a for a, s in score.items() if s >= 3)
print(f"\nFlagged {len(flagged)} accounts across 2 rings: {', '.join(flagged)}")
print(f"Shared-device cluster with no other signal: {sorted(m for m in by_device['dev_FAM'])}"
      f" -> REVIEW (2 points), not FLAG")
```
{% endraw %}

Running it produces:

```text
=== Shared-device clusters (soft link) ===
  dev_FAM   -> acct_F1, acct_F3   [2 accounts]
  dev_RINGA -> acct_A1, acct_A2, acct_A3, acct_A4   [4 accounts]

=== Shared-phone components (hard identity link) ===
  acct_A1 + acct_A2 + acct_A3 + acct_A4
  acct_B1 + acct_B2 + acct_B3

=== Money-flow cycles (layering loops, <=3 hops) ===
  acct_B1 -> acct_B2 -> acct_B3 -> acct_B1   (KSh 190,000-200,000 per leg)

=== Ring scores (device + identity + loop + flow shape) ===
  account       device  phone  loop  thru  fan  score  verdict
  acct_A1            +      +     -     +    +      5  FLAG
  acct_A2            +      +     -     +    -      4  FLAG
  acct_A3            +      +     -     +    -      4  FLAG
  acct_A4            +      +     -     +    -      4  FLAG
  acct_B1            -      +     +     +    -      3  FLAG
  acct_B2            -      +     +     +    -      3  FLAG
  acct_B3            -      +     +     +    -      3  FLAG
  acct_F1            +      -     -     -    -      2  REVIEW
  acct_F3            +      -     -     -    -      2  REVIEW

Flagged 7 accounts across 2 rings: acct_A1, acct_A2, acct_A3, acct_A4, acct_B1, acct_B2, acct_B3
Shared-device cluster with no other signal: ['acct_F1', 'acct_F3'] -> REVIEW (2 points), not FLAG
```

Read the output the way a fraud analyst would:

- **Two rings surface through two different doors.** Ring A is caught by *soft + hard links* — four accounts on one device and one phone, fanning out; Ring B is caught by its *loop*, with no shared device at all. A system watching only devices misses B; one watching only loops misses A. No legitimate payment reaches the table.
- **The household cluster lands at REVIEW, not FLAG.** A shared device alone is worth two points — enough to watch, not to accuse. This is the method's most important design choice: **flag on corroboration, not a single weak signal.** Every real deployment drowns in households, offices and cybercafés that share devices.

## From heuristics to GNNs

Hand-written rules are the right first version: every threshold is explainable to an investigator and an auditor. Next, let a model learn the pattern. Graph features — PageRank, degree centrality, local clustering coefficient, k-core membership — are computed from the same graph and fed into a conventional tabular model; Neo4j notes that "a device used by many high-value cards will have a high PageRank, signaling potential risk" ([Neo4j](https://neo4j.com/developer/industry-use-cases/finserv/retail-banking/ieee-cis-fraud-graphs/)).

Beyond features sit graph neural networks — GraphSAGE, GCN, R-GCN and attention-based variants — which propagate information across neighbouring nodes so a prediction about one account depends on the company it keeps. The public proving ground is the **Elliptic** Bitcoin graph: **203,769 nodes and 234,355 edges**, of which only **4,545 (about 2%) are labelled illicit** ([Nature Scientific Reports](https://www.nature.com/articles/s41598-025-95672-w), [Kumo](https://kumo.ai/pyg/datasets/elliptic-bitcoin/)).

Two cautions. First, imbalance is brutal: at ~2% positives, precision at a usable recall is the metric that matters, and a false positive is a blocked legitimate customer — Aite-Novarica found that **almost 90% of declined transactions are legitimate** ([reported by Analytics Vidhya](https://www.analyticsvidhya.com/blog/2025/11/gnn-fraud-detection-with-neo4j/)). Second, graphs are **adversarially malleable**: a ring that suspects it is being clustered can stop sharing devices, rotate numbers, add hops and split into smaller, cleaner components. Monitoring cluster size and link density over time is how you notice the graph being gamed.

## How we can do better

1. **Build the entity graph first.** You cannot reconstruct rings from a flat transaction table. Persist accounts, devices, phones, cards and national IDs as nodes with typed edges from day one — hard-link components are your cheapest high-precision signal.
2. **Run connected components on every refresh.** Union-find over shared identity links is linear and explainable, and collapses a ring into one entity to investigate instead of forty unrelated alerts.
3. **Treat community and cycle detection as ring detectors.** Louvain/Leiden finds the dense "fraud island" distinct from the legitimate giant component; directed cycle detection catches layering loops no per-account score will see.
4. **Score rings, not accounts, and route to ring-level review.** One case with all members, links and narrative is what an investigator can work — and what a court needs.
5. **Demand corroboration before flagging.** Weight independent signals so shared-device-only clusters stay at review, as the demo does: shared device *plus* phone *plus* fan-out *plus* a loop is a case; a shared device alone is a watchlist.
6. **Share signals across institutions.** A ring spanning three banks and a wallet is invisible to each one alone — precisely why EMMA and Jackal IV are cross-border, and why sector data-sharing and fast FRC reporting break the second hop.

## Key takeaways

| The lesson | Why it matters |
|---|---|
| The unit of detection is the ring, not the account | Members are individually unremarkable; the signal is in the edges |
| Hard links give precision, soft links give coverage | Collapse identity components first, then cluster behaviour (25M → 7.7M nodes) |
| Louvain finds "fraud islands"; cycle detection finds layering | Two rings in the demo surfaced through two different doors |
| Corroboration beats any single signal | Shared-device-only clusters are usually families, not fraud |
| At a ~2% positive rate, watch precision-at-recall | Aite-Novarica: ~90% of declined transactions are legitimate |
| Graphs can be gamed by splitting | Monitor cluster size and link density as an adversarial signal |

## References

- [OCCRP — Global Operation Targets Money Mules (Europol EMMA 9)](https://www.occrp.org/en/news/global-operation-targets-money-mules)
- [BBC — West African cyber-crime networks: mass arrests follow Interpol crackdown (Operation Jackal IV)](https://www.bbc.com/news/articles/cq5xdnxppl4o)
- [Sahara Reporters — Operation Jackal: INTERPOL arrests 58, identifies 260+ suspects](https://saharareporters.com/2026/08/25/operation-jackal-interpol-arrests-58-persons-identifies-over-260-west-african-organised)
- [Cifas — Fraudscape 2026](https://www.cifas.org.uk/newsroom/fraudscape2026)
- [Neo4j — A Graph-Based Approach to Financial Fraud Detection (IEEE-CIS)](https://neo4j.com/developer/industry-use-cases/finserv/retail-banking/ieee-cis-fraud-graphs/)
- [arXiv:2512.19061 — Fraud Detection Through Large-Scale Graph Clustering with Heterogeneous Link Transformation](https://arxiv.org/abs/2512.19061)
- [arXiv:2411.05815 — Graph Neural Networks for Financial Fraud Detection: A Review](https://arxiv.org/abs/2411.05815)
- [Nature Scientific Reports — Graph convolution network for fraud detection in bitcoin transactions (Elliptic)](https://www.nature.com/articles/s41598-025-95672-w)
- [Kumo — Elliptic Bitcoin Dataset (203,769 nodes, 234,355 edges)](https://kumo.ai/pyg/datasets/elliptic-bitcoin/)
- [Analytics Vidhya — Building a Real-Time Fraud Detection System with GNNs and Neo4j](https://www.analyticsvidhya.com/blog/2025/11/gnn-fraud-detection-with-neo4j/)

## Related posts

- [Fraud ML in Mobile Money: velocity, graph features and the 70-account loophole](/posts/fraud-ml-mobile-money/)
- [One Bank, Many Buckets: Anatomy of the Sidian Bank Sh7.88M Mule Heist](/posts/sidian-bank-mule-heist/)
- [Fraud Models Rot Quietly: PSI Drift and Data-Quality Gates](/posts/fraud-model-drift-monitoring/)
- [Graph Algorithms in Neo4j: PageRank and Community Detection](/posts/graph-algorithms-neo4j/)
- [Anomaly Detection for Reconciliation at Scale](/posts/anomaly-detection-reconciliation/)
