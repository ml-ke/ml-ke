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
- Faded neural network (3-layer MLP with 4-3-2 nodes, opacity 0.15)
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

---

## 11. Agent Fundamentals — ReAct Cycle (Observe→Think→Act)
**File:** `cover-agent-fundamentals.svg`
**SVG Elements:**
- Central brain outline (two mirrored cerebral hemispheres, purple)
- Outer circular arrow cycle: 4 curved Bezier arrows forming a ring (observe → think → act → observe)
- Observation node (top-left): eye/radar icon with cyan rays
- Thought node (top-right): brain hemisphere with purple glow
- Action node (bottom): gear with green glow
- Node labels: "OBSERVE", "REASON", "ACT" positioned at each arrow segment
- Small dotted return path showing the feedback loop
- Color: cyan (#00d2ff) observation, purple (#a78bfa) reasoning, green (#6bcf7f) action

---

## 12. Agent Memory Systems — Three-Tiered Storage Pyramid
**File:** `cover-agent-memory-systems.svg`
**SVG Elements:**
- 3-tier pyramid structure, each tier a labeled block:
  - Top: buffer (conversation window, smallest) — cyan
  - Middle: vector store (embeddings, medium) — purple
  - Bottom: disk (SQLite persistent, widest) — green
- Arrow paths between tiers (up for retrieval, down for storage)
- Magnifying glass icon on top tier with "working memory" label
- Database icon on bottom tier with "persistent" label
- Flow arrows: left side data-in, right side query-in, top response-out
- Color: cyan (#00d2ff) buffer, purple (#a78bfa) vectors, green (#6bcf7f) disk

---

## 13. Tool Calling — Robot Arm + Tool Icons
**File:** `cover-agent-tool-calling.svg`
**SVG Elements:**
- Robot arm: 2-segment articulated arm (upper arm + forearm), joint circles
- End effector (gripper) reaching toward tool icons
- 3 tool icons in a row:
  - Wrench (cyan) — utility/configuration
  - Magnifying glass (purple) — search/knowledge
  - Calculator (green) — math/logic
- Tool call bubbles: small rectangles above each tool showing function call syntax
- Small checkmarks on completed tools, pulsing ring on active tool
- Color: cyan (#00d2ff) wrench, purple (#a78bfa) magnifier, green (#6bcf7f) calculator

---

## 14. Multi-Agent Systems — Orbiting Agent Nodes + Supervisor
**File:** `cover-multi-agent-systems.svg`
**SVG Elements:**
- Large central supervisor node (pulsing purple circle, 2x size)
- 4 orbiting agent nodes connected to supervisor by solid white lines
- Each agent node: circle with unique accent color and icon
  - Researcher (cyan, magnifier icon)
  - Writer (green, text icon)
  - Reviewer (yellow, checkmark icon)
  - Executor (red, gear icon)
- Dashed lines between agents for lateral communication
- Outer ring orbit paths (dotted ellipses)
- Labels: "SUPERVISOR" center, agent names below each node
- Color: purple (#a78bfa) supervisor, multi-color agents

---

## 15. Agent Observability — Magnifying Glass + Log Stream Trace
**File:** `cover-agent-observability.svg`
**SVG Elements:**
- Large magnifying glass (circle + handle) positioned over log stream
- Log stream: 5 horizontal colored bars (green/amber/red) with timestamps
- Trace spans: nested horizontal bars showing parent→child span relationships
- Key metrics callouts: "LLM: 2.3s", "Search: 0.8s", "Memory: 0.1s" with latency bars
- Cross-cutting trace ID linking log entries
- Warning flags: red markers on slow spans
- Color: green (#6bcf7f) success, amber (#ffd93d) warning, red (#ff6b6b) error

---

## 16. Agent Production Deployment — Pipeline Assembly Line
**File:** `cover-agent-production-deployment.svg`
**SVG Elements:**
- Conveyor belt: horizontal path with small tick marks moving left to right
- 4 rectangular stages on the belt: DEV → TEST → STAGE → PROD
- Each stage: colored block with upward arrow (promotion gate)
- Agent packages: small boxes on the conveyor between stages
- Redis queue icon (cylinder) at intake
- Load balancer (circle with dividing arrows) at production end
- Metrics alongside: "p50: 120ms", "p99: 450ms" with sparkline trend
- Color: green (#6bcf7f) dev, cyan (#00d2ff) test, amber (#ffd93d) stage, purple (#a78bfa) prod

---

## 17. Agent Security — Shield + Gear + Lock
**File:** `cover-agent-security.svg`
**SVG Elements:**
- Large shield shape (chevron + rounded base) taking center frame
- Inside shield: gear mechanism with 6 teeth, interlocking
- Padlock icon superimposed on gear: rect + shackle arc + keyhole
- Firewall bars: 3 vertical bars with alternating open/closed indicators on left side
- Input validation gate: funnel shape on top narrowing into shield
- Sandbox container: dashed bounding box at bottom with "SANDBOX" label
- Color: cyan (#00d2ff) shield outline, green (#6bcf7f) lock, yellow (#ffd93d) gate

---

## 18. African AI Landscape — Continent Outline + Circuit Nodes
**File:** `cover-african-ai-landscape.svg`
**SVG Elements:**
- African continent outline (simplified path, thick stroke, white fill)
- Inside continent: 6 glowing tech nodes positioned at major cities
  - North: Cairo hexagon
  - West: Lagos triangle
  - East: Nairobi circle
  - South: Cape Town hexagon
  - Central: Kinshasa pentagon
- Circuit traces connecting nodes (right-angle paths, glowing cyan)
- Small network rings radiating from each node (concentric circles, fading)
- External data arrows entering continent from top and sides
- "AI IN AFRICA" label below continent
- Color: green (#6bcf7f) continent fill, cyan (#00d2ff) circuits, purple (#a78bfa) nodes

---

## 19. Swahili NLP — Speech Bubbles + Translation Arrows
**File:** `cover-swahili-nlp.svg`
**SVG Elements:**
- 3 speech bubbles in a diagonal cascade:
  - Left: large bubble with Swahili text "Habari ya leo?" in purple
  - Center: translation arrow transforming bubble contents
  - Right: large bubble with English "How are you today?" in cyan
- Translation engine icon between bubbles: gear with language codes SW→EN
- Tokenization diagram below: "Habari ya leo?" split into [Habar] [i] [ya] [leo] [?] tiles
- Small Swahili words scattered as decorative elements: "asante", "rafiki", "ndoto", "mwalimu"
- Bottom bar: code snippet showing Swahili BERT tokenizer output
- Color: purple (#a78bfa) Swahili, cyan (#00d2ff) English, green (#6bcf7f) engine

---

## 20. Low-Resource NLP — Seedling Growing from Limited Data
**File:** `cover-low-resource-nlp.svg`
**SVG Elements:**
- Small seed at bottom: oval shape with crack line, tiny root below
- Small data droplets (5-6 small circles) falling onto seed from above
- Growing stem: curved green line rising from seed
- Branching leaves: 3 leaf shapes at different heights, getting larger
- Leaf veins: thin lines within leaves
- Blooming flower at top: multi-petal flower with ML icon at center
- Data augmentation symbols: curved arrows showing back-translation (EN→SW→EN), word substitution circles
- Transfer learning arrow: large downward arrow from "pre-trained XLM-R" cloud to seed
- Color: green (#6bcf7f) plant, cyan (#00d2ff) data, purple (#a78bfa) augmentation

---

## 21. Mobile-First AI — Phone + Brain + Network
**File:** `cover-mobile-first-ai.svg`
**SVG Elements:**
- Phone outline: rounded rect with screen, camera notch, and home indicator
- Inside screen: brain/neural icon (simplified node + connections)
- Network waves radiating outward from phone: 4 concentric semicircular arcs
- TFLite icon: small lightning bolt in bottom-left corner of screen
- Model quantization callouts: "FP32→INT8" with shrinking arrow
- Offline badge: "✓ OFFLINE" text with checkmark
- Edge compute labels: "on-device", "zero-latency", "privacy-first"
- Color: cyan (#00d2ff) phone, purple (#a78bfa) AI brain, green (#6bcf7f) offline

---

## 22. African AI Communities — Connected Network Map
**File:** `cover-african-ai-communities.svg`
**SVG Elements:**
- Stylized map of Africa (simplified outline, faint)
- 5 community node clusters over key locations:
  - Masakhane (Nairobi/South Africa) — purple, NLP symbol
  - Deep Learning Indaba (pan-African) — cyan, conference icon
  - Data Science Nigeria (Lagos) — green, DS icon
  - Lacuna Fund (regional) — yellow, data icon
  - Local meetup nodes (smaller) — gray
- Connection lines between all nodes forming a mesh network
- People icons (simplified stick figures) attached to each node
- Collaboration arrows: bidirectional between Masakhane and Indaba
- Data flow paths from Lacuna Fund to other nodes (dotted, gold)
- Color: purple (#a78bfa) Masakhane, cyan (#00d2ff) Indaba, green (#6bcf7f) DSNigeria, yellow (#ffd93d) Lacuna

---

## 23. AI for Agriculture — Leaf Circuit + Satellite
**File:** `cover-ai-for-agriculture.svg`
**SVG Elements:**
- Large leaf outline filling center, divided into left/right halves
- Left half: organic leaf veins (natural branching paths, green)
- Right half: circuit traces (right-angle PCB traces, cyan) — nature meets tech
- Crop row patterns at bottom: parallel dashed lines representing farm rows
- Satellite dish in top-right: parabolic arc + feed horn + signal waves
- Crop disease detection overlay: small dotted red circle on one leaf section
- Weather icons: sun, rain cloud, temperature (small symbols in a row)
- Yield prediction graph at bottom: upward-trending line with data points
- Color: green (#6bcf7f) leaf, cyan (#00d2ff) circuits, red (#ff6b6b) detection

---

## 24. Constrained Environment AI — Edge Device + Offline
**File:** `cover-constrained-environment-ai.svg`
**SVG Elements:**
- Edge device: small rectangular box with antenna, LEDs, and ports
- Offline symbol: circle with diagonal slash over a signal bar icon
- Battery icon: rect with charge level indicator (2/3 full, green)
- TinyML chip: small IC with "TINY" label and 8 pins
- Model compression arrows: large model → quantization/pruning → tiny model
- Sync-when-available icon: cloud with dotted connection, arrow indicating periodic sync
- Low-power indicator: small crescent moon symbol
- Data flow: thin dotted line from edge device to cloud (occasional), thick solid local processing path
- Color: green (#6bcf7f) device, cyan (#00d2ff) compression, amber (#ffd93d) offline warning

---

## 25. Model Serving 101 — Notebook → Server Rack
**File:** `cover-model-serving-101.svg`
**SVG Elements:**
- Left side: Jupyter notebook icon (stack of pages, Python logo, "train.ipynb" label)
- Center: transformation arrow (large curved arrow with "EXPORT" label)
- Right side: server rack (3-tier block with LED indicators and cooling vents)
- API endpoints emanating from server: GET /predict, POST /batch, GET /health
- Docker container icons (whale + boxes) floating between notebook and server
- Request flow: client→load balancer→server with response times "45ms", "120ms"
- Model format badges: "ONNX ✓", "TorchScript ✓", "SavedModel ✓"
- Color: cyan (#00d2ff) notebook, purple (#a78bfa) transformation, green (#6bcf7f) server

---

## 26. vLLM High-Throughput Serving — Parallel Request Engine
**File:** `cover-vllm-llm-serving.svg`
**SVG Elements:**
- 4 parallel request arrows entering left side (different colors for different clients)
- vLLM engine: large gear/engine icon center, labeled "vLLM ENGINE"
- Inside engine: KV cache diagram showing PagedAttention blocks (small page rectangles with page table pointers)
- Continuous batching: multiple request tokens being processed simultaneously (interleaved token sequences)
- Response streams exiting right side: 4 parallel green streams
- Performance annotations: "Tokens/s: 3200", "p50 TTFT: 200ms" with small sparkline
- Quantization callouts: "FP16→INT4" with memory reduction arrow "4x smaller"
- Color: multi-color requests, purple (#a78bfa) engine, green (#6bcf7f) responses

---

## 27. GPU Optimization — GPU Die + Core Grid
**File:** `cover-gpu-optimization.svg`
**SVG Elements:**
- GPU chip die: large central rectangle with labeled functional blocks
- SM (Streaming Multiprocessor) grid: 6x4 matrix of small squares representing compute cores
- Memory hierarchy:
  - HBM2 stack: 4 vertical blocks on sides, labeled "HBM2 80GB"
  - L2 cache: narrow band between SMs and HBM
- Heat zones: color gradient across the die (cool blue center → warm red edges)
- Cooling indicators: small fan icons at top and bottom
- Tensor core highlighting: specific SM squares glowing purple
- Memory bandwidth annotation: "2 TB/s" with arrow
- Color: cyan (#00d2ff) cool, red (#ff6b6b) hot, purple (#a78bfa) tensor cores

---

## 28. Kubernetes ML — Helm Ship Wheel + Orbiting Model Pods
**File:** `cover-kubernetes-ml.svg`
**SVG Elements:**
- Center: K8s helm/ship wheel with 8 spokes and central hub
- 5 model container pods orbiting the wheel on elliptical paths
- Each pod: rectangular container with model name label, positioned on orbit
  - "BERT API" (cyan), "GPT API" (purple), "ResNet" (green), "YOLO" (amber), "Whisper" (red)
- Auto-scaling indicators: horizontal arrows showing pods scaling up (right arrow) and down (left arrow)
- GPU scheduling marker: "GPU: Tesla T4" label on one pod
- Kserve InferenceService: small CRD manifest snippet floating near one pod
- Service mesh paths: dotted lines between pods showing traffic routing
- Prometheus monitoring nodes: small circles with "P8s" label collecting metrics
- Color: cyan (#00d2ff) BERT, purple (#a78bfa) GPT, green (#6bcf7f) ResNet, amber (#ffd93d) YOLO

---

## 29. ML CI/CD — Pipeline Stages + Feedback Loop
**File:** `cover-ml-cicd.svg`
**SVG Elements:**
- Horizontal pipeline with 5 stage blocks connected by arrows:
  1. DATA (database icon, cyan) → Data validation gate (diamond shape)
  2. TRAIN (GPU icon, purple) → Model checkpoint (disk icon)
  3. EVAL (chart icon, amber) → Metric threshold check (pass/fail indicator)
  4. DEPLOY (rocket icon, green) → Canary deployment marker (traffic split %)
  5. MONITOR (dashboard icon, red) → Drift detection alarm
- Circular CI/CD feedback arrow: large arc from MONITOR back to DATA (continuous improvement loop)
- Model registry between EVAL and DEPLOY: MLflow icon with version labels (v1.0, v1.1, v1.2)
- GitHub Actions badge: small workflow icon on each stage
- Rollback path: dashed red reverse arrow from DEPLOY back to EVAL
- Color: cyan → purple → amber → green → red stage flow

---

## 30. ML Monitoring — Dashboard + Drift + Alerts
**File:** `cover-ml-monitoring.svg`
**SVG Elements:**
- 3-panel dashboard layout:
  - Panel 1 (left, large): Gauge meter with needle in yellow zone (slight drift detected)
  - Panel 2 (right-top): Drift distribution curves — reference (blue bell curve) and current (red shifted curve) with arrow showing shift
  - Panel 3 (right-bottom): Timeline sparkline showing metric over time with red alert spike
- Alert indicators: red pulsing dots on the dashboard with "ALERT" badges
- Metric labels: "Data Drift: 0.23 (p<0.01)", "Accuracy: 94.2%", "Latency p99: 210ms"
- Evidently AI dashboard style: clean card panels with white on dark
- Model version annotation: "Model v1.2 → deployed 2026-06-01 — status: MONITORING"
- Slack/teams notification icon: small chat bubble with alert symbol
- Color: green (#6bcf7f) healthy, amber (#ffd93d) warning, red (#ff6b6b) alert
