"""
Generate all SVG cover images for the ML Kenya blog series.
Each cover is 1200x630 with dark tech aesthetic and unique per-topic styling.
"""

import os

COVERS = {
    "kg-neo4j": {
        "title": ["Building a Knowledge Graph", "with Neo4j and Python"],
        "subtitle": "KNOWLEDGE GRAPHS · HANDS-ON TUTORIAL",
        "nodes": "graph"
    },
    "kg-embeddings": {
        "title": ["Knowledge Graph Embeddings:", "From TransE to RotatE"],
        "subtitle": "KGE · VECTOR REPRESENTATIONS",
        "nodes": "vectors"
    },
    "kg-gnn": {
        "title": ["Graph Neural Networks", "for Knowledge Graph Reasoning"],
        "subtitle": "GNNS · RELATIONAL LEARNING",
        "nodes": "neural"
    },
    "kg-production": {
        "title": ["Knowledge Graphs", "in Production"],
        "subtitle": "SCALING · STORAGE · OPTIMIZATION",
        "nodes": "infra"
    },
    "kg-llm": {
        "title": ["Knowledge Graphs Meet LLMs:", "RAG with Structured Knowledge"],
        "subtitle": "LLM · RETRIEVAL-AUGMENTED GENERATION",
        "nodes": "hybrid"
    },
    "ai-prompt-injection": {
        "title": ["Prompt Injection:", "The #1 LLM Security Risk"],
        "subtitle": "OWASP LLM TOP 10 · LLM01",
        "nodes": "danger"
    },
    "ai-jailbreak": {
        "title": ["Jailbreaking LLMs:", "From DAN to GODMODE"],
        "subtitle": "ADVERSARIAL PROMPTING · RED TEAMING",
        "nodes": "exploit"
    },
    "ai-data-poisoning": {
        "title": ["Data Poisoning and", "Model Backdoors"],
        "subtitle": "TRAINING-TIME ATTACKS · SUPPLY CHAIN",
        "nodes": "poison"
    },
    "ai-agent-security": {
        "title": ["Insecure Agent Design:", "When AI Has Too Much Agency"],
        "subtitle": "AGENT SECURITY · EXCESSIVE AGENCY",
        "nodes": "agent"
    },
    "ai-supply-chain": {
        "title": ["Supply Chain Attacks", "on AI Systems"],
        "subtitle": "MODEL REQUEST FORGERY · ECOSYSTEM RISKS",
        "nodes": "chain"
    },
    # === 10 NEW IDEAS ===
    "agent-skills-evolution": {
        "title": ["Self-Evolving Agent Skills:", "When AI Rewrites Its Playbook"],
        "subtitle": "AUTONOMOUS SKILL EVOLUTION · SECURITY AUDIT",
        "nodes": "neural"
    },
    "memory-attack-surface": {
        "title": ["Memory as an Attack Surface:", "Poisoning AI Agent Memory"],
        "subtitle": "PERSISTENT MEMORY · AGENT SECURITY",
        "nodes": "graph"
    },
    "automated-science-guardrails": {
        "title": ["Guardrails for Autonomous", "AI Research Pipelines"],
        "subtitle": "AI-DRIVEN SCIENCE · AUDIT · REPRODUCIBILITY",
        "nodes": "hybrid"
    },
    "multi-agent-collusion": {
        "title": ["Multi-Agent Collusion:", "When AI Agents Conspire"],
        "subtitle": "COORDINATION · SAFETY · DETECTION",
        "nodes": "agent"
    },
    "structured-data-injection": {
        "title": ["Prompt Injection via", "Structured Data Formats"],
        "subtitle": "PDF · JSON · CSV · DATABASE RECORDS",
        "nodes": "exploit"
    },
    "automated-red-teaming": {
        "title": ["Automated AI Red Teaming", "at Scale"],
        "subtitle": "VULNERABILITY DISCOVERY · LLM SECURITY",
        "nodes": "danger"
    },
    "model-watermarking": {
        "title": ["Model Watermarking:", "Proving AI Ownership"],
        "subtitle": "THEFT DETECTION · IP PROTECTION",
        "nodes": "vectors"
    },
    "eval-benchmark-poisoning": {
        "title": ["Eval Benchmark Poisoning:", "Gaming the Leaderboards"],
        "subtitle": "DATA CONTAMINATION · BENCHMARK INTEGRITY",
        "nodes": "poison"
    },
    "multimodal-attacks": {
        "title": ["Real-Time Multimodal", "Security: Audio/Video Attacks"],
        "subtitle": "VOICE INJECTION · VIDEO PROMPTS · OMNIMODAL",
        "nodes": "neural"
    },
    "ml-secrets-management": {
        "title": ["ML Pipeline Secrets", "Management"],
        "subtitle": "VAULT · API KEYS · CREDENTIAL HYGIENE",
        "nodes": "infra"
    },
}

def make_svg(key, data):
    title1, title2 = data["title"]
    subtitle = data["subtitle"]
    
    # Color themes per node style
    themes = {
        "graph": {"primary": "#6bcf7f", "secondary": "#00d2ff", "accent": "#ffd93d"},
        "vectors": {"primary": "#00d2ff", "secondary": "#6bcf7f", "accent": "#a78bfa"},
        "neural": {"primary": "#a78bfa", "secondary": "#00d2ff", "accent": "#6bcf7f"},
        "infra": {"primary": "#ffd93d", "secondary": "#ff6b6b", "accent": "#00d2ff"},
        "hybrid": {"primary": "#ffd93d", "secondary": "#6bcf7f", "accent": "#00d2ff"},
        "danger": {"primary": "#ff6b6b", "secondary": "#ffd93d", "accent": "#ff4444"},
        "exploit": {"primary": "#ff4444", "secondary": "#ff6b6b", "accent": "#ffd93d"},
        "poison": {"primary": "#ff6b6b", "secondary": "#a78bfa", "accent": "#ffd93d"},
        "agent": {"primary": "#ffd93d", "secondary": "#ff6b6b", "accent": "#00d2ff"},
        "chain": {"primary": "#a78bfa", "secondary": "#ff6b6b", "accent": "#00d2ff"}
    }
    c = themes[data["nodes"]]
    p, s, a = c["primary"], c["secondary"], c["accent"]

    svg_nodes = ""
    if data["nodes"] == "graph":
        svg_nodes = f'''
  <g filter="url(#glow)">
    <circle cx="600" cy="315" r="70" fill="none" stroke="url(#g1)" stroke-width="2" opacity="0.6"/>
    <circle cx="600" cy="315" r="15" fill="{p}" opacity="0.8"/>
    <circle cx="400" cy="220" r="10" fill="{s}" opacity="0.7"/>
    <circle cx="800" cy="240" r="10" fill="{s}" opacity="0.7"/>
    <circle cx="350" cy="400" r="10" fill="{a}" opacity="0.7"/>
    <circle cx="750" cy="430" r="10" fill="{a}" opacity="0.7"/>
    <circle cx="480" cy="160" r="8" fill="{p}" opacity="0.5"/>
    <circle cx="720" cy="170" r="8" fill="{p}" opacity="0.5"/>
    <circle cx="300" cy="300" r="8" fill="{a}" opacity="0.5"/>
    <circle cx="850" cy="320" r="8" fill="{a}" opacity="0.5"/>
    <circle cx="200" cy="150" r="4" fill="{s}" opacity="0.3"/>
    <circle cx="950" cy="180" r="4" fill="{s}" opacity="0.3"/>
  </g>
  <g stroke="{p}" stroke-width="1" opacity="0.15">
    <line x1="600" y1="315" x2="400" y2="220"/>
    <line x1="600" y1="315" x2="800" y2="240"/>
    <line x1="600" y1="315" x2="350" y2="400"/>
    <line x1="600" y1="315" x2="750" y2="430"/>
    <line x1="400" y1="220" x2="480" y2="160"/>
    <line x1="800" y1="240" x2="720" y2="170"/>
  </g>'''
    elif data["nodes"] == "vectors":
        svg_nodes = f'''
  <g filter="url(#glow)">
    <rect x="250" y="250" width="100" height="100" rx="8" fill="none" stroke="{p}" stroke-width="2" opacity="0.5"/>
    <rect x="450" y="220" width="100" height="100" rx="8" fill="none" stroke="{s}" stroke-width="2" opacity="0.5"/>
    <rect x="650" y="240" width="100" height="100" rx="8" fill="none" stroke="{a}" stroke-width="2" opacity="0.5"/>
    <rect x="850" y="260" width="100" height="100" rx="8" fill="none" stroke="{p}" stroke-width="2" opacity="0.4"/>
    <text x="300" y="305" text-anchor="middle" font-family="monospace" font-size="12" fill="{p}" opacity="0.7">h₁</text>
    <text x="500" y="275" text-anchor="middle" font-family="monospace" font-size="12" fill="{s}" opacity="0.7">h₂</text>
    <text x="700" y="295" text-anchor="middle" font-family="monospace" font-size="12" fill="{a}" opacity="0.7">h₃</text>
    <text x="900" y="315" text-anchor="middle" font-family="monospace" font-size="12" fill="{p}" opacity="0.6">hₙ</text>
  </g>
  <g stroke="{p}" stroke-width="1.5" opacity="0.15">
    <line x1="350" y1="300" x2="450" y2="270"/>
    <line x1="550" y1="270" x2="650" y2="290"/>
    <line x1="750" y1="290" x2="850" y2="310"/>
  </g>'''
    elif data["nodes"] == "neural":
        svg_nodes = f'''
  <g filter="url(#glow)">
    <!-- Input layer -->
    <circle cx="300" cy="200" r="8" fill="{p}" opacity="0.7"/>
    <circle cx="300" cy="280" r="8" fill="{p}" opacity="0.7"/>
    <circle cx="300" cy="360" r="8" fill="{p}" opacity="0.7"/>
    <circle cx="300" cy="440" r="8" fill="{p}" opacity="0.7"/>
    <!-- Hidden layer -->
    <circle cx="500" cy="180" r="8" fill="{s}" opacity="0.7"/>
    <circle cx="500" cy="260" r="8" fill="{s}" opacity="0.7"/>
    <circle cx="500" cy="340" r="8" fill="{s}" opacity="0.7"/>
    <circle cx="500" cy="420" r="8" fill="{s}" opacity="0.7"/>
    <circle cx="500" cy="500" r="8" fill="{s}" opacity="0.7"/>
    <!-- Hidden layer 2 -->
    <circle cx="700" cy="200" r="8" fill="{a}" opacity="0.7"/>
    <circle cx="700" cy="280" r="8" fill="{a}" opacity="0.7"/>
    <circle cx="700" cy="360" r="8" fill="{a}" opacity="0.7"/>
    <circle cx="700" cy="440" r="8" fill="{a}" opacity="0.7"/>
    <!-- Output -->
    <circle cx="900" cy="260" r="10" fill="{p}" opacity="0.8"/>
    <circle cx="900" cy="380" r="10" fill="{s}" opacity="0.8"/>
  </g>
  <g stroke="{p}" stroke-width="0.5" opacity="0.12">
    <line x1="308" y1="200" x2="492" y2="180"/>
    <line x1="308" y1="200" x2="492" y2="260"/>
    <line x1="308" y1="280" x2="492" y2="260"/>
    <line x1="308" y1="280" x2="492" y2="340"/>
    <line x1="308" y1="360" x2="492" y2="340"/>
    <line x1="308" y1="360" x2="492" y2="420"/>
    <line x1="308" y1="440" x2="492" y2="420"/>
    <line x1="308" y1="440" x2="492" y2="500"/>
    <line x1="508" y1="180" x2="692" y2="200"/>
    <line x1="508" y1="260" x2="692" y2="200"/>
    <line x1="508" y1="260" x2="692" y2="280"/>
    <line x1="508" y1="340" x2="692" y2="280"/>
    <line x1="508" y1="340" x2="692" y2="360"/>
    <line x1="508" y1="420" x2="692" y2="360"/>
    <line x1="508" y1="420" x2="692" y2="440"/>
    <line x1="508" y1="500" x2="692" y2="440"/>
    <line x1="708" y1="200" x2="892" y2="260"/>
    <line x1="708" y1="280" x2="892" y2="260"/>
    <line x1="708" y1="280" x2="892" y2="380"/>
    <line x1="708" y1="360" x2="892" y2="380"/>
    <line x1="708" y1="440" x2="892" y2="380"/>
  </g>'''
    elif data["nodes"] == "infra":
        svg_nodes = f'''
  <g filter="url(#glow)">
    <rect x="250" y="200" width="180" height="80" rx="6" fill="none" stroke="{p}" stroke-width="2" opacity="0.5"/>
    <rect x="270" y="220" width="60" height="40" rx="4" fill="{p}" opacity="0.3"/>
    <rect x="340" y="220" width="70" height="40" rx="4" fill="{s}" opacity="0.3"/>
    <rect x="550" y="200" width="180" height="80" rx="6" fill="none" stroke="{s}" stroke-width="2" opacity="0.5"/>
    <rect x="570" y="220" width="60" height="40" rx="4" fill="{s}" opacity="0.3"/>
    <rect x="640" y="220" width="70" height="40" rx="4" fill="{a}" opacity="0.3"/>
    <rect x="750" y="320" width="180" height="80" rx="6" fill="none" stroke="{a}" stroke-width="2" opacity="0.5"/>
    <rect x="770" y="340" width="140" height="40" rx="4" fill="{a}" opacity="0.3"/>
    <text x="340" y="245" text-anchor="middle" font-family="monospace" font-size="10" fill="#e6edf3" opacity="0.6">DB</text>
    <text x="375" y="245" text-anchor="middle" font-family="monospace" font-size="10" fill="#e6edf3" opacity="0.6">CACHE</text>
    <text x="600" y="245" text-anchor="middle" font-family="monospace" font-size="10" fill="#e6edf3" opacity="0.6">API</text>
    <text x="675" y="245" text-anchor="middle" font-family="monospace" font-size="10" fill="#e6edf3" opacity="0.6">INDEX</text>
    <text x="840" y="362" text-anchor="middle" font-family="monospace" font-size="10" fill="#e6edf3" opacity="0.6">QUERY ENGINE</text>
  </g>
  <g stroke="{p}" stroke-width="1.5" opacity="0.2">
    <line x1="430" y1="240" x2="550" y2="240"/>
    <line x1="730" y1="240" x2="840" y2="340"/>
  </g>'''
    elif data["nodes"] == "hybrid":
        svg_nodes = f'''
  <g filter="url(#glow)">
    <!-- Graph half -->
    <circle cx="350" cy="280" r="12" fill="{p}" opacity="0.8"/>
    <circle cx="250" cy="220" r="8" fill="{s}" opacity="0.6"/>
    <circle cx="430" cy="210" r="8" fill="{s}" opacity="0.6"/>
    <circle cx="280" cy="360" r="8" fill="{a}" opacity="0.6"/>
    <circle cx="440" cy="350" r="8" fill="{a}" opacity="0.6"/>
    <!-- LLM half -->
    <rect x="700" y="200" width="160" height="100" rx="10" fill="none" stroke="{p}" stroke-width="2" opacity="0.5"/>
    <text x="780" y="255" text-anchor="middle" font-family="monospace" font-size="14" fill="{p}" opacity="0.7">LLM</text>
    <rect x="720" y="360" width="120" height="60" rx="8" fill="none" stroke="{s}" stroke-width="2" opacity="0.4"/>
    <text x="780" y="395" text-anchor="middle" font-family="monospace" font-size="10" fill="{s}" opacity="0.6">RAG</text>
  </g>
  <g stroke="{a}" stroke-width="1.5" opacity="0.25" stroke-dasharray="6,4">
    <line x1="500" y1="280" x2="700" y2="250"/>
  </g>'''
    elif data["nodes"] == "danger":
        svg_nodes = f'''
  <g filter="url(#glow)">
    <!-- Warning triangle -->
    <polygon points="600,150 350,480 850,480" fill="none" stroke="{p}" stroke-width="3" opacity="0.5"/>
    <polygon points="600,210 450,440 750,440" fill="none" stroke="{a}" stroke-width="1" opacity="0.3"/>
    <text x="600" y="370" text-anchor="middle" font-size="40" fill="{p}" opacity="0.7">!</text>
    <circle cx="400" cy="280" r="6" fill="{a}" opacity="0.5"/>
    <circle cx="750" cy="300" r="6" fill="{a}" opacity="0.5"/>
    <circle cx="500" cy="240" r="4" fill="{s}" opacity="0.4"/>
    <circle cx="680" cy="250" r="4" fill="{s}" opacity="0.4"/>
  </g>
  <g stroke="{p}" stroke-width="1" opacity="0.12">
    <line x1="400" y1="280" x2="500" y2="240"/>
    <line x1="400" y1="280" x2="600" y2="210"/>
    <line x1="750" y1="300" x2="680" y2="250"/>
    <line x1="750" y1="300" x2="600" y2="210"/>
  </g>'''
    elif data["nodes"] == "exploit":
        svg_nodes = f'''
  <g filter="url(#glow)">
    <rect x="300" y="200" width="180" height="100" rx="8" fill="none" stroke="{p}" stroke-width="2" opacity="0.4"/>
    <text x="390" y="255" text-anchor="middle" font-family="monospace" font-size="11" fill="{p}" opacity="0.6">USER INPUT</text>
    <rect x="600" y="200" width="180" height="100" rx="8" fill="none" stroke="{s}" stroke-width="2" opacity="0.4"/>
    <text x="690" y="255" text-anchor="middle" font-family="monospace" font-size="11" fill="{s}" opacity="0.6">LLM</text>
    <rect x="450" y="380" width="180" height="80" rx="8" fill="none" stroke="{a}" stroke-width="2" opacity="0.4"/>
    <text x="540" y="428" text-anchor="middle" font-family="monospace" font-size="11" fill="{a}" opacity="0.6">OUTPUT</text>
    <!-- Arrows -->
    <polyline points="480,250 590,250" fill="none" stroke="{p}" stroke-width="2" opacity="0.5" marker-end="none"/>
    <polygon points="585,243 600,250 585,257" fill="{p}" opacity="0.5"/>
    <polyline points="690,300 540,375" fill="none" stroke="{a}" stroke-width="2" opacity="0.5"/>
    <polygon points="546,380 536,369 550,370" fill="{a}" opacity="0.5"/>
  </g>'''
    elif data["nodes"] == "poison":
        svg_nodes = f'''
  <g filter="url(#glow)">
    <!-- Dataset -->
    <rect x="250" y="200" width="200" height="120" rx="6" fill="none" stroke="{p}" stroke-width="2" opacity="0.4"/>
    <text x="350" y="230" text-anchor="middle" font-family="monospace" font-size="10" fill="{p}" opacity="0.6">DATASET</text>
    <!-- Poison needle -->
    <rect x="300" y="260" width="10" height="40" rx="2" fill="{a}" opacity="0.8" transform="rotate(-15,305,280)"/>
    <circle cx="302" cy="262" r="4" fill="{a}" opacity="0.9"/>
    <!-- Model -->
    <rect x="550" y="200" width="200" height="120" rx="10" fill="none" stroke="{s}" stroke-width="2" opacity="0.4"/>
    <text x="650" y="230" text-anchor="middle" font-family="monospace" font-size="10" fill="{s}" opacity="0.6">TRAINED MODEL</text>
    <!-- Backdoor trigger in model -->
    <circle cx="630" cy="280" r="8" fill="none" stroke="{a}" stroke-width="1.5" opacity="0.6"/>
    <text x="630" y="283" text-anchor="middle" font-size="8" fill="{a}" opacity="0.7">⚡</text>
  </g>
  <g stroke="{s}" stroke-width="1.5" opacity="0.3">
    <line x1="450" y1="260" x2="550" y2="260"/>
  </g>'''
    elif data["nodes"] == "agent":
        svg_nodes = f'''
  <g filter="url(#glow)">
    <rect x="200" y="180" width="150" height="80" rx="8" fill="none" stroke="{p}" stroke-width="2" opacity="0.4"/>
    <text x="275" y="222" text-anchor="middle" font-family="monospace" font-size="10" fill="{p}" opacity="0.6">AGENT</text>
    <rect x="450" y="140" width="130" height="60" rx="6" fill="none" stroke="{s}" stroke-width="2" opacity="0.4"/>
    <text x="515" y="175" text-anchor="middle" font-family="monospace" font-size="10" fill="{s}" opacity="0.6">TOOL A</text>
    <rect x="450" y="240" width="130" height="60" rx="6" fill="none" stroke="{s}" stroke-width="2" opacity="0.4"/>
    <text x="515" y="275" text-anchor="middle" font-family="monospace" font-size="10" fill="{s}" opacity="0.6">TOOL B</text>
    <rect x="450" y="340" width="130" height="60" rx="6" fill="none" stroke="{a}" stroke-width="2" opacity="0.4"/>
    <text x="515" y="375" text-anchor="middle" font-family="monospace" font-size="10" fill="{a}" opacity="0.6">DATA</text>
    <!-- Red alert -->
    <circle cx="720" cy="280" r="20" fill="none" stroke="{p}" stroke-width="2" opacity="0.6"/>
    <text x="720" y="285" text-anchor="middle" font-size="18" fill="{p}" opacity="0.8">⚠</text>
  </g>
  <g stroke="{p}" stroke-width="1" opacity="0.2">
    <line x1="350" y1="210" x2="450" y2="170"/>
    <line x1="350" y1="220" x2="450" y2="270"/>
    <line x1="350" y1="230" x2="450" y2="370"/>
    <line x1="580" y1="170" x2="700" y2="270"/>
    <line x1="580" y1="270" x2="700" y2="280"/>
  </g>'''
    elif data["nodes"] == "chain":
        svg_nodes = f'''
  <g filter="url(#glow)">
    <!-- Chain of boxes -->
    <rect x="150" y="250" width="120" height="60" rx="6" fill="none" stroke="{p}" stroke-width="2" opacity="0.4"/>
    <text x="210" y="285" text-anchor="middle" font-family="monospace" font-size="10" fill="{p}" opacity="0.6">REPO</text>
    <rect x="330" y="250" width="120" height="60" rx="6" fill="none" stroke="{s}" stroke-width="2" opacity="0.4"/>
    <text x="390" y="285" text-anchor="middle" font-family="monospace" font-size="10" fill="{s}" opacity="0.6">CI/CD</text>
    <rect x="510" y="250" width="120" height="60" rx="6" fill="none" stroke="{a}" stroke-width="2" opacity="0.4"/>
    <text x="570" y="285" text-anchor="middle" font-family="monospace" font-size="10" fill="{a}" opacity="0.6">DEPLOY</text>
    <rect x="690" y="250" width="120" height="60" rx="6" fill="none" stroke="{p}" stroke-width="2" opacity="0.4"/>
    <text x="750" y="285" text-anchor="middle" font-family="monospace" font-size="10" fill="{p}" opacity="0.6">MODEL</text>
    <!-- Chain links -->
    <g stroke="{a}" stroke-width="2" opacity="0.3" fill="none">
      <ellipse cx="280" cy="265" rx="8" ry="4"/>
      <ellipse cx="280" cy="275" rx="8" ry="4"/>
      <line x1="272" y1="265" x2="272" y2="275"/>
      <line x1="288" y1="265" x2="288" y2="275"/>
      <ellipse cx="460" cy="265" rx="8" ry="4"/>
      <ellipse cx="460" cy="275" rx="8" ry="4"/>
      <line x1="452" y1="265" x2="452" y2="275"/>
      <line x1="468" y1="265" x2="468" y2="275"/>
      <ellipse cx="640" cy="265" rx="8" ry="4"/>
      <ellipse cx="640" cy="275" rx="8" ry="4"/>
      <line x1="632" y1="265" x2="632" y2="275"/>
      <line x1="648" y1="265" x2="648" y2="275"/>
    </g>
    <!-- Broken link -->
    <line x1="810" y1="260" x2="840" y2="300" stroke="{p}" stroke-width="2" opacity="0.4" stroke-dasharray="4,3"/>
  </g>'''
    
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 630" width="1200" height="630">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0d1117"/>
      <stop offset="50%" style="stop-color:#161b22"/>
      <stop offset="100%" style="stop-color:#1a1a2e"/>
    </linearGradient>
    <linearGradient id="g1" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:{p}"/>
      <stop offset="100%" style="stop-color:{s}"/>
    </linearGradient>
    <filter id="glow">
      <feGaussianBlur stdDeviation="3" result="b"/>
      <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
    </filter>
  </defs>
  <rect width="1200" height="630" fill="url(#bg)"/>
  <g opacity="0.03" stroke="#fff" stroke-width="0.5">
    <line x1="0" y1="70" x2="1200" y2="70"/><line x1="0" y1="140" x2="1200" y2="140"/>
    <line x1="0" y1="210" x2="1200" y2="210"/><line x1="0" y1="280" x2="1200" y2="280"/>
    <line x1="0" y1="350" x2="1200" y2="350"/><line x1="0" y1="420" x2="1200" y2="420"/>
    <line x1="0" y1="490" x2="1200" y2="490"/><line x1="0" y1="560" x2="1200" y2="560"/>
    <line x1="150" y1="0" x2="150" y2="630"/><line x1="300" y1="0" x2="300" y2="630"/>
    <line x1="450" y1="0" x2="450" y2="630"/><line x1="600" y1="0" x2="600" y2="630"/>
    <line x1="750" y1="0" x2="750" y2="630"/><line x1="900" y1="0" x2="900" y2="630"/>
    <line x1="1050" y1="0" x2="1050" y2="630"/>
  </g>
  {svg_nodes}
  <text x="600" y="140" text-anchor="middle" font-family="'Segoe UI','Helvetica Neue',Arial,sans-serif" font-size="34" font-weight="700" fill="#e6edf3" letter-spacing="1">{title1}</text>
  <text x="600" y="182" text-anchor="middle" font-family="'Segoe UI','Helvetica Neue',Arial,sans-serif" font-size="34" font-weight="700" fill="#e6edf3" letter-spacing="1">{title2}</text>
  <text x="600" y="520" text-anchor="middle" font-family="'Segoe UI','Helvetica Neue',Arial,sans-serif" font-size="13" fill="#8b949e" letter-spacing="3">{subtitle}</text>
  <line x1="420" y1="535" x2="780" y2="535" stroke="url(#g1)" stroke-width="2" opacity="0.4"/>
  <text x="1090" y="600" text-anchor="end" font-family="'Segoe UI','Helvetica Neue',Arial,sans-serif" font-size="11" fill="#484f58">ml-ke.github.io</text>
  <circle cx="1110" cy="607" r="3" fill="{p}" opacity="0.6"/>
</svg>'''

for key, data in COVERS.items():
    path = f"/home/pro-g/ProG/ml-ke/assets/blog/cover-{key}.svg"
    with open(path, "w") as f:
        f.write(make_svg(key, data))
    print(f"Created {path}")

print("\nAll 10 cover images generated!")
