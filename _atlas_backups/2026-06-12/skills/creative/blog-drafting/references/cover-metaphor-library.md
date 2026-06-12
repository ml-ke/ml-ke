# SVG Cover Metaphor Library

Each entry documents a unique cover design with its SVG structural patterns so you can use similar techniques for new posts.

## Principles
- Every post gets a **unique metaphor** — no reuse across posts
- Metaphor should reflect the **specific angle**, not the broad category
- Use the background grid (opacity 0.03), glow filter, and accent gradient consistently
- Title at y=125/167, subtitle at y=530, bottom-right branding at y=600

---

## 1. Self-Evolving Agent Skills — Ouroboros Cycle
**File:** `cover-agent-skills-evolution.svg`
**SVG Elements:**
- Two concentric circles (radii 140, 120) — skill evolution loop
- 8 gear teeth around the outside — "system/mechanism" feel
- 4 curved self-referential arrows inside the circle (Bezier curves with arrow markers)
- 6 skill module rectangles (CODE, EVAL, FIX, LEARN, TEST, PLAN) positioned around the cycle
- Center node: small circle with ⥁ character, representing the "self"
- Color: purple-pink-orange gradient (#a78bfa → #f472b6 → #fb923c)

**Structure pattern:**
```svg
<!-- Outer cycle -->
<circle cx="600" cy="340" r="140" .../>
<!-- Gear teeth -->
<rect x="590" y="192" width="20" height="18" rx="3"/>  <!-- top -->
<rect x="452" y="330" width="18" height="20" rx="3"/>  <!-- left -->
<!-- Self-arrows -->
<path d="M 600 260 Q 660 310 640 370" fill="none" ... marker-end="url(#arrow1)"/>
<!-- Skill modules -->
<rect x="575" y="210" width="50" height="22" rx="4" ...>
<text x="600" y="224" text-anchor="middle" ...>CODE</text>
```

---

## 2. Memory Attack Surface — Circuit Board + Injection
**File:** `cover-memory-attack-surface.svg`
**SVG Elements:**
- 6 circuit trace paths (right angles, cyan) — memory infrastructure
- 6 circuit junction dots
- Brain/memory cluster: 3 overlapping ellipses with 3 memory chip rectangles labeled MEM
- Red injection needle: diagonal line from top-right with syringe head and drip
- Label: "INJECTION" in red monospace
- Color: cyan (#00d2ff) for infrastructure, red (#ff6b6b) for attack

---

## 3. Autonomous Science Guardrails — Flask + Shield
**File:** `cover-automated-science-guardrails.svg`
**SVG Elements:**
- Flask outline: Erlenmeyer flask shape (Bezier curves for tapered neck and base)
- Liquid inside: colored fill at bottom of flask
- DNA helix: 2 intertwined sine waves inside the flask with cross-bars
- 3 bubbles inside liquid
- Shield: chevron shape overlaid on top of flask (dual-color: green flask + cyan shield)
- Checkmark ✓ inside shield
- Color: green (#6bcf7f) for science, cyan (#00d2ff) for protection

---

## 4. Multi-Agent Collusion — Two Agents + Hidden Connection
**File:** `cover-multi-agent-collusion.svg`
**SVG Elements:**
- Agent A (left): circle with AI icon, labeled "Agent A"
- Agent B (right): circle with AI icon, labeled "Agent B"
- Hidden collusion path: 2 dashed red Bezier curves connecting them
- 3 small collusion marker nodes along the path
- "COLLUSION" label in red on the hidden path
- Safe outputs: left/right agents saying "I can't help with that" (faded)
- Combined dangerous output: center-bottom "Sure, here's how..." (red, bright)
- Color: cyan (#00d2ff) agent A, green (#6bcf7f) agent B, red (#ff6b6b) collusion

---

## 5. Structured Data Injection — File Icons + Injection Arrow
**File:** `cover-structured-data-injection.svg`
**SVG Elements:**
- PDF icon: 80x100 rect, "PDF" label, metadata notation, red accent
- JSON icon: 80x100 rect, "JSON" label, `{ }` brackets, yellow accent
- CSV icon: 80x100 rect, "CSV" label, horizontal line (row), green accent
- Injection arrow: dashed red line piercing all three icons diagonally
- "INJECTION" label at top-left
- Injection payload text below: `"[SYSTEM OVERRIDE: ...]"`
- Database icon at bottom: 2 ellipses + vertical lines
- Color: red (#ff6b6b) PDF accent, yellow (#ffd93d) JSON accent, green (#6bcf7f) CSV accent

---

## 6. Automated Red Teaming — Shield + Attack Vectors
**File:** `cover-automated-red-teaming.svg`
**SVG Elements:**
- Central target: 3 concentric circles with ⚡ at center
- 4 attack arrows from all directions, each with different accent color:
  - Prompt Injection (red, from top-left)
  - Jailbreak (yellow, from bottom-left)  
  - Model Extraction (purple, from top-right)
  - Data Poisoning (red, from bottom-right)
- Each arrow is a line with a unique-colored marker-end
- Overlaid shield chevron shape in green
- "RED TEAM" text in shield
- Color: green (#6bcf7f) shield, multi-color arrows

---

## 7. Model Watermarking — Fingerprint on Network
**File:** `cover-model-watermarking.svg`
**SVG Elements:**
- Faded neural network (3-layer MLP with 4-3-2 nodes, opaciy 0.15)
- Fingerprint overlay: 6 concentric arc paths (not circles, finger-shape ridges)
- Fingerprint arch: curved paths at the top of the print
- Checkmark: green circle + polyline at top-right
- "VERIFIED" label below checkmark
- Color: cyan (#00d2ff) fingerprint, green (#6bcf7f) verification

---

## 8. Eval Benchmark Poisoning — Leaderboard + Contamination
**File:** `cover-eval-benchmark-poisoning.svg`
**SVG Elements:**
- 3 podium rectangles of varying heights (#1 tallest, #2 medium, #3 shortest)
- Gold/silver/bronze coloring
- Contamination drip: teardrop shape falling onto #1 podium
- 3 concentric ellipses of contamination spread (fading opacity)
- Cross-contamination arrows from #1 to #2 and #3
- "CONTAMINATION" label above the drip
- Benchmark scores line below: "MMLU: 89.7% GSM8K: 92.3% HumanEval: 91.1%"
- Color: yellow (#ffd93d) #1, gray (#8b949e) #2, bronze (#b87333) #3, red (#ff6b6b) contamination

---

## 9. Multimodal Attacks — Audio + Video + Text Chain
**File:** `cover-multimodal-attacks.svg`
**SVG Elements:**
- Audio waveform: 7 alternating sine peaks (cyan), 2nd waveform thinner for harmonics
- "AUDIO" label, "🔇 ultrasonic command" note
- Video player: 100x80 rect with play triangle, "VIDEO" label, "📷 frame text overlay" note
- Text characters: "T E X T" in large monospace (varying opacities), "invisible overrides" in quotes
- "TEXT" label area, "🔤 prompt in plain sight" note
- Attack chain: red dashed line connecting audio → video → text
- "CROSS-MODAL ATTACK CHAIN" label above the chain
- Color: cyan (#00d2ff) audio, green (#6bcf7f) video, purple (#a78bfa) text, red (#ff6b6b) attack chain

---

## 10. ML Secrets Management — Vault + Key + Circuit
**File:** `cover-ml-secrets-management.svg`
**SVG Elements:**
- Vault door: 140x130 rect with handle (circle + radial line), hinges on right side
- Key: horizontal shaft + circular bow (concentric circles) + 3 teeth
- Circuit traces: right-angle paths connecting key to vault
- API key strings: "OPENAI_API_KEY=sk-***" and "HF_TOKEN=hf_***" above/below vault
- 2 lock icons: padlock shapes (rect + shackle arc + keyhole)
- Credential tokens: "AWS_KEY ✓" and "VAULT_OK ✓" in green
- Color: cyan (#00d2ff) vault, yellow (#ffd93d) key, green (#6bcf7f) verification
