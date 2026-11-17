# Meta-Analysis Workflow: The 10+ Iteration Deep Dive (Added Jun 2026)

## When to Use

When the user pushes back on a shallow conclusion with "that's not always true" or "that's
a bit lacking." This is the signal that surface-level pattern-matching failed and we need
to go deeper.

## The Problem

Our default mode is: find a pattern → declare a rule → move on. This produces conclusions
that are technically correct but miss nuance. Example from this session:

**Shallow pass**: "We prove mechanisms, not exploits. The fix is to always prove exploits."
**User**: "I find your conclusion a bit lacking. This is not always true."
**Deep pass**: After 10+ research iterations across CVEs, disclosed reports, top hunters'
writeups, and our own rejection history → identified 5 distinct root causes, not one.

## The 10+ Iteration Process

DO NOT stop at iteration 1. Each iteration must uncover something the previous one missed.

### Iteration Template

```
Iteration N:
  1. LEARN — Search a DIFFERENT source type than the last iteration
     (Iter 1: PortSwigger → Iter 2: H1 Hacktivity → Iter 3: CVEs → Iter 4: Writeups → ...)
  2. EXTRACT — What specific findings/techniques work? What don't?
  3. COMPARE — How does this differ from what I found in iteration N-1?
  4. SYNTHESIZE — What's the pattern across ALL iterations so far?
  5. IDENTIFY GAPS — What am I still missing? What contradicts my emerging theory?
  6. REPEAT
```

### Iteration Source Rotation (at minimum, hit each once)

| Iteration | Source | What to Extract |
|-----------|--------|-----------------|
| 1 | Our own rejection history | Why each finding failed. Map to gate. |
| 2 | CVE database (same vuln class) | What made accepted CVEs different from our rejections? |
| 3 | H1 Hacktivity (disclosed reports) | What titles, impact language, PoC formats do accepted reports share? |
| 4 | Top hunter writeups (Shubham Shah, etc.) | What methodology do they use that we don't? |
| 5 | PortSwigger research | What detection/prevention techniques exist? |
| 6 | Bugcrowd/Intigriti blog | What do programs say they want? |
| 7 | Program scope pages | What are the actual acceptance criteria? |
| 8 | Compare across vuln classes | Is this pattern universal or class-specific? |
| 9 | Test against a live target | Does the corrected theory hold? |
| 10 | Synthesize | What's the complete, nuanced answer? |

### Example: This Session's Meta-Analysis

After 10+ iterations across CVEs (Composio, WordPress OAuth, pac4j JWT),
disclosed reports (DoD VDP, UPS VDP, top-100 H1 auth reports), top hunter
methodology (Shubham Shah/Assetnote), and our own rejection history:

**Correct conclusion**: Not "we prove mechanisms, not exploits." But rather:

1. **We hunt for architecture, not security boundaries** — MCP case: two-layer
   system where gateway leniency is design, not bug.
2. **We hunt for labels, not actions** — Our titles describe the mechanism, not
   what an attacker can DO.
3. **We don't understand what "impact" means to the program** — Competitive
   intelligence is explicitly excluded. Data must enable a DIRECT action.
4. **We skip the architecture research phase** — We probe endpoints before
   understanding the security boundary model.
5. **We don't use the right information sources** — Shodan, Censys, GitHub
   dorking, OIDC configs are underutilized.

### Pitfall: Stopping at One Source

If all your examples come from the same source (e.g., only H1 reports, or only
our own rejections), you haven't done enough iterations. Force rotation:

- "What does a CVE database say about this?"
- "What does a top hunter say about this?"
- "What does a program owner say about this?"
- "What does a live test say about this?"
- "What does a DIFFERENT vulnerability class say about this?"

### The Shallow Conclusion Detector

Before presenting any meta-analysis, ask yourself:

- Does this conclusion fit on a bumper sticker? ("We prove mechanisms, not exploits")
- Can I name 3 specific counterexamples from different sources?
- Did I check at least 5 different source types?
- Did I find a contradiction that forced me to refine my theory?
- If yes to all of the above → probably a good conclusion. Present with nuance.
- If no to any → go back to iteration.
