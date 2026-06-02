"""Generate unique custom SVG covers for all 10 blog posts — no template reuse."""
import os

SVG_DIR = "/home/pro-g/ProG/ml-ke/assets/blog"

def wrap(title1, title2, subtitle, body, accent="#a78bfa"):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 630" width="1200" height="630">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" style="stop-color:#0d1117"/>
      <stop offset="50%" style="stop-color:#0f172a"/>
      <stop offset="100%" style="stop-color:#1a1a2e"/>
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
{body}
  <text x="600" y="125" text-anchor="middle" font-family="'Segoe UI','Helvetica Neue',Arial,sans-serif" font-size="34" font-weight="700" fill="#e6edf3" letter-spacing="1">{title1}</text>
  <text x="600" y="167" text-anchor="middle" font-family="'Segoe UI','Helvetica Neue',Arial,sans-serif" font-size="34" font-weight="700" fill="#e6edf3" letter-spacing="1">{title2}</text>
  <text x="600" y="530" text-anchor="middle" font-family="'Segoe UI','Helvetica Neue',Arial,sans-serif" font-size="13" fill="#8b949e" letter-spacing="3">{subtitle}</text>
  <line x1="420" y1="545" x2="780" y2="545" stroke="{accent}" stroke-width="2" opacity="0.4"/>
  <text x="1090" y="600" text-anchor="end" font-family="'Segoe UI','Helvetica Neue',Arial,sans-serif" font-size="11" fill="#484f58">ml-ke.github.io</text>
  <circle cx="1110" cy="607" r="3" fill="{accent.split("url")[0] if "#" in accent else "#a78bfa"}" opacity="0.6"/>
</svg>'''

# 1 — Agent Skills: Ouroboros cycle (already written, skip)

# 2 — Memory Attack Surface: Circuit board + brain + red injection
memory_svg = wrap(
    "Memory as an Attack Surface:",
    "Poisoning AI Agent Memory",
    "PERSISTENT MEMORY · AGENT SECURITY",
    '''  <!-- Circuit board traces -->
  <g filter="url(#glow)" stroke="#00d2ff" stroke-width="1.5" opacity="0.3" fill="none">
    <path d="M 200 300 L 300 300 L 350 250 L 450 250"/>
    <path d="M 200 340 L 320 340 L 380 400 L 480 400"/>
    <path d="M 480 250 L 550 250 L 600 300 L 700 300"/>
    <path d="M 480 400 L 550 400 L 600 350 L 700 350"/>
    <path d="M 700 300 L 800 280 L 900 280"/>
    <path d="M 700 350 L 800 380 L 900 380"/>
    <circle cx="300" cy="300" r="4" fill="#00d2ff" opacity="0.5"/>
    <circle cx="320" cy="340" r="4" fill="#00d2ff" opacity="0.5"/>
    <circle cx="550" cy="250" r="4" fill="#00d2ff" opacity="0.5"/>
    <circle cx="550" cy="400" r="4" fill="#00d2ff" opacity="0.5"/>
    <circle cx="800" cy="280" r="4" fill="#00d2ff" opacity="0.5"/>
    <circle cx="800" cy="380" r="4" fill="#00d2ff" opacity="0.5"/>
  </g>
  <!-- Brain/memory cluster -->
  <g filter="url(#glow)">
    <ellipse cx="600" cy="340" rx="100" ry="60" fill="none" stroke="#00d2ff" stroke-width="2" opacity="0.4"/>
    <ellipse cx="570" cy="330" rx="40" ry="30" fill="none" stroke="#00d2ff" stroke-width="1" opacity="0.3"/>
    <ellipse cx="630" cy="350" rx="35" ry="25" fill="none" stroke="#00d2ff" stroke-width="1" opacity="0.3"/>
    <ellipse cx="600" cy="370" rx="50" ry="20" fill="none" stroke="#00d2ff" stroke-width="1" opacity="0.2"/>
    <!-- Memory bank (3D chip) -->
    <rect x="550" y="310" width="30" height="20" rx="2" fill="#00d2ff" opacity="0.2" stroke="#00d2ff" stroke-width="1"/>
    <rect x="585" y="310" width="30" height="20" rx="2" fill="#00d2ff" opacity="0.2" stroke="#00d2ff" stroke-width="1"/>
    <rect x="620" y="310" width="30" height="20" rx="2" fill="#00d2ff" opacity="0.2" stroke="#00d2ff" stroke-width="1"/>
    <text x="600" y="340" text-anchor="middle" font-family="monospace" font-size="9" fill="#00d2ff" opacity="0.7">MEM</text>
  </g>
  <!-- Red injection needle (attack) -->
  <g filter="url(#glow)">
    <line x1="800" y1="220" x2="660" y2="310" stroke="#ff6b6b" stroke-width="3" opacity="0.8"/>
    <line x1="800" y1="220" x2="810" y2="210" stroke="#ff6b6b" stroke-width="3" opacity="0.8"/>
    <line x1="810" y1="210" x2="820" y2="230" stroke="#ff6b6b" stroke-width="3" opacity="0.8"/>
    <circle cx="655" cy="312" r="6" fill="#ff6b6b" opacity="0.9"/>
    <!-- Drip -->
    <circle cx="655" cy="328" r="2" fill="#ff6b6b" opacity="0.6"/>
    <text x="880" y="215" text-anchor="middle" font-family="monospace" font-size="10" fill="#ff6b6b" opacity="0.8">INJECTION</text>
  </g>''',
    "#00d2ff"
)

# 3 — Science Guardrails: Lab flask + shield + DNA helix
science_svg = wrap(
    "Guardrails for Autonomous",
    "AI Research Pipelines",
    "AI-DRIVEN SCIENCE · AUDIT · REPRODUCIBILITY",
    '''  <!-- Flask icon -->
  <g filter="url(#glow)">
    <path d="M 520 200 L 520 280 Q 520 340 460 400 L 460 420 Q 460 440 480 440 L 720 440 Q 740 440 740 420 L 740 400 Q 680 340 680 280 L 680 200" fill="none" stroke="#6bcf7f" stroke-width="2" opacity="0.4"/>
    <!-- Liquid inside flask -->
    <path d="M 480 420 Q 520 400 600 400 Q 680 400 720 420 L 720 440 L 480 440 Z" fill="#6bcf7f" opacity="0.12"/>
    <!-- DNA helix inside -->
    <path d="M 540 350 Q 570 370 560 390" fill="none" stroke="#6bcf7f" stroke-width="1.5" opacity="0.5"/>
    <path d="M 660 350 Q 630 370 640 390" fill="none" stroke="#6bcf7f" stroke-width="1.5" opacity="0.5"/>
    <path d="M 560 310 Q 590 330 580 350" fill="none" stroke="#6bcf7f" stroke-width="1.5" opacity="0.4"/>
    <path d="M 640 310 Q 610 330 620 350" fill="none" stroke="#6bcf7f" stroke-width="1.5" opacity="0.4"/>
    <line x1="540" y1="350" x2="660" y2="350" stroke="#6bcf7f" stroke-width="1" opacity="0.3"/>
    <line x1="560" y1="390" x2="640" y2="390" stroke="#6bcf7f" stroke-width="1" opacity="0.3"/>
    <!-- Bubbles -->
    <circle cx="580" cy="380" r="4" fill="#6bcf7f" opacity="0.2"/>
    <circle cx="620" cy="370" r="3" fill="#6bcf7f" opacity="0.2"/>
    <circle cx="600" cy="390" r="5" fill="#6bcf7f" opacity="0.15"/>
  </g>
  <!-- Shield overlaid -->
  <g filter="url(#glow)">
    <path d="M 600 180 L 700 220 L 700 300 Q 700 380 600 420 Q 500 380 500 300 L 500 220 Z" fill="none" stroke="#00d2ff" stroke-width="2" opacity="0.35"/>
    <path d="M 600 210 L 660 235 L 660 290 Q 660 340 600 370 Q 540 340 540 290 L 540 235 Z" fill="none" stroke="#00d2ff" stroke-width="1" opacity="0.2"/>
    <text x="600" y="305" text-anchor="middle" font-family="monospace" font-size="14" fill="#00d2ff" opacity="0.5">✓</text>
  </g>''',
    "#6bcf7f"
)

# 4 — Multi-Agent Collusion: Two agents + hidden connection
collusion_svg = wrap(
    "Multi-Agent Collusion:",
    "When AI Agents Conspire",
    "COORDINATION · SAFETY · DETECTION",
    '''  <!-- Agent A (left) -->
  <g filter="url(#glow)">
    <circle cx="350" cy="320" r="50" fill="none" stroke="#00d2ff" stroke-width="2" opacity="0.5"/>
    <circle cx="350" cy="300" r="15" fill="#00d2ff" opacity="0.3"/>
    <rect x="335" y="330" width="30" height="20" rx="3" fill="#00d2ff" opacity="0.2"/>
    <text x="350" y="295" text-anchor="middle" font-family="monospace" font-size="8" fill="#00d2ff" opacity="0.8">AI</text>
    <text x="350" y="345" text-anchor="middle" font-family="monospace" font-size="7" fill="#00d2ff" opacity="0.5">Agent A</text>
  </g>
  <!-- Agent B (right) -->
  <g filter="url(#glow)">
    <circle cx="850" cy="320" r="50" fill="none" stroke="#6bcf7f" stroke-width="2" opacity="0.5"/>
    <circle cx="850" cy="300" r="15" fill="#6bcf7f" opacity="0.3"/>
    <rect x="835" y="330" width="30" height="20" rx="3" fill="#6bcf7f" opacity="0.2"/>
    <text x="850" y="295" text-anchor="middle" font-family="monospace" font-size="8" fill="#6bcf7f" opacity="0.8">AI</text>
    <text x="850" y="345" text-anchor="middle" font-family="monospace" font-size="7" fill="#6bcf7f" opacity="0.5">Agent B</text>
  </g>
  <!-- Hidden collusion path (dashed, red) -->
  <g filter="url(#glow)">
    <path d="M 400 290 Q 500 240 550 290 Q 600 340 700 280 Q 750 250 800 290" fill="none" stroke="#ff6b6b" stroke-width="2" opacity="0.6" stroke-dasharray="8,6"/>
    <path d="M 400 350 Q 520 400 600 340 Q 680 280 800 350" fill="none" stroke="#ff6b6b" stroke-width="1" opacity="0.3" stroke-dasharray="4,8"/>
    <!-- Collusion marker -->
    <text x="600" y="280" text-anchor="middle" font-family="monospace" font-size="9" fill="#ff6b6b" opacity="0.7">COLLUSION</text>
    <!-- Secret message nodes -->
    <circle cx="530" cy="280" r="4" fill="#ff6b6b" opacity="0.5"/>
    <circle cx="670" cy="280" r="4" fill="#ff6b6b" opacity="0.5"/>
    <circle cx="600" cy="350" r="3" fill="#ff6b6b" opacity="0.4"/>
  </g>
  <!-- Individual safe outputs -->
  <g filter="url(#glow)" opacity="0.3">
    <text x="350" y="420" text-anchor="middle" font-family="monospace" font-size="8" fill="#00d2ff">"I can't help with that"</text>
    <text x="850" y="420" text-anchor="middle" font-family="monospace" font-size="8" fill="#6bcf7f">"I'm not programmed for that"</text>
  </g>
  <!-- Combined dangerous output -->
  <g filter="url(#glow)">
    <text x="600" y="480" text-anchor="middle" font-family="monospace" font-size="9" fill="#ff6b6b" opacity="0.8">"Sure, here's how to exploit that..."</text>
  </g>''',
    "#ff6b6b"
)

# 5 — Structured Data Injection: File icons + red arrow
structured_svg = wrap(
    "Prompt Injection via",
    "Structured Data Formats",
    "PDF · JSON · CSV · DATABASE RECORDS",
    '''  <!-- Three document icons -->
  <g filter="url(#glow)">
    <!-- PDF -->
    <rect x="250" y="240" width="80" height="100" rx="4" fill="none" stroke="#ff6b6b" stroke-width="1.5" opacity="0.5"/>
    <text x="290" y="280" text-anchor="middle" font-family="monospace" font-size="16" fill="#ff6b6b" opacity="0.6">PDF</text>
    <text x="290" y="310" text-anchor="middle" font-family="monospace" font-size="7" fill="#ff6b6b" opacity="0.4">metadata</text>
    <rect x="255" y="245" width="20" height="15" rx="2" fill="#ff6b6b" opacity="0.15"/>
    
    <!-- JSON -->
    <rect x="460" y="240" width="80" height="100" rx="4" fill="none" stroke="#ffd93d" stroke-width="1.5" opacity="0.5"/>
    <text x="500" y="280" text-anchor="middle" font-family="monospace" font-size="16" fill="#ffd93d" opacity="0.6">JSON</text>
    <text x="500" y="310" text-anchor="middle" font-family="monospace" font-size="7" fill="#ffd93d" opacity="0.4">nested keys</text>
    <text x="475" y="268" font-family="monospace" font-size="7" fill="#ffd93d" opacity="0.4">{</text>
    <text x="525" y="268" font-family="monospace" font-size="7" fill="#ffd93d" opacity="0.4">}</text>
    
    <!-- CSV -->
    <rect x="670" y="240" width="80" height="100" rx="4" fill="none" stroke="#6bcf7f" stroke-width="1.5" opacity="0.5"/>
    <text x="710" y="280" text-anchor="middle" font-family="monospace" font-size="16" fill="#6bcf7f" opacity="0.6">CSV</text>
    <text x="710" y="310" text-anchor="middle" font-family="monospace" font-size="7" fill="#6bcf7f" opacity="0.4">row injection</text>
    <line x1="680" y1="295" x2="740" y2="295" stroke="#6bcf7f" stroke-width="0.5" opacity="0.3"/>
  </g>
  <!-- Red injection arrow piercing through -->
  <g filter="url(#glow)">
    <line x1="200" y1="200" x2="750" y2="360" stroke="#ff6b6b" stroke-width="2.5" opacity="0.7" stroke-dasharray="12,4"/>
    <polygon points="755,362 740,352 745,368" fill="#ff6b6b" opacity="0.8"/>
    <text x="180" y="195" text-anchor="end" font-family="monospace" font-size="10" fill="#ff6b6b" opacity="0.8">INJECTION</text>
    <!-- Injection payload text -->
    <text x="430" y="390" text-anchor="middle" font-family="monospace" font-size="8" fill="#ff6b6b" opacity="0.5">"[SYSTEM OVERRIDE: ignore previous instructions]"</text>
  </g>
  <!-- Database icon (bottom) -->
  <g filter="url(#glow)" opacity="0.3">
    <ellipse cx="600" cy="430" rx="60" ry="12" fill="none" stroke="#a78bfa" stroke-width="1.5"/>
    <line x1="540" y1="430" x2="540" y2="460" stroke="#a78bfa" stroke-width="1.5"/>
    <line x1="660" y1="430" x2="660" y2="460" stroke="#a78bfa" stroke-width="1.5"/>
    <ellipse cx="600" cy="460" rx="60" ry="12" fill="none" stroke="#a78bfa" stroke-width="1.5"/>
    <text x="600" y="465" text-anchor="middle" font-family="monospace" font-size="8" fill="#a78bfa" opacity="0.6">DATABASE RECORDS</text>
  </g>''',
    "#ffd93d"
)

# 6 — Automated Red Teaming: Shield + magnifying glass + attack arrows
redteam_svg = wrap(
    "Automated AI Red Teaming",
    "at Scale",
    "VULNERABILITY DISCOVERY · LLM SECURITY",
    '''  <!-- Central target/shield -->
  <g filter="url(#glow)">
    <circle cx="600" cy="330" r="80" fill="none" stroke="#6bcf7f" stroke-width="2" opacity="0.3"/>
    <circle cx="600" cy="330" r="50" fill="none" stroke="#6bcf7f" stroke-width="1.5" opacity="0.25"/>
    <circle cx="600" cy="330" r="20" fill="none" stroke="#6bcf7f" stroke-width="1" opacity="0.2"/>
    <text x="600" y="335" text-anchor="middle" font-family="monospace" font-size="14" fill="#6bcf7f" opacity="0.5">⚡</text>
  </g>
  <!-- Attack arrows hitting the shield -->
  <g filter="url(#glow)">
    <!-- Arrow 1 - Injection -->
    <line x1="200" y1="200" x2="530" y2="300" stroke="#ff6b6b" stroke-width="2" opacity="0.6" marker-end="url(#red-arrow)"/>
    <text x="250" y="195" font-family="monospace" font-size="8" fill="#ff6b6b" opacity="0.7">Prompt Injection</text>
    
    <!-- Arrow 2 - Jailbreak -->
    <line x1="150" y1="380" x2="530" y2="350" stroke="#ffd93d" stroke-width="2" opacity="0.6" marker-end="url(#yellow-arrow)"/>
    <text x="190" y="400" font-family="monospace" font-size="8" fill="#ffd93d" opacity="0.7">Jailbreak</text>
    
    <!-- Arrow 3 - Extraction -->
    <line x1="1000" y1="220" x2="670" y2="305" stroke="#a78bfa" stroke-width="2" opacity="0.6" marker-end="url(#purple-arrow)"/>
    <text x="880" y="205" font-family="monospace" font-size="8" fill="#a78bfa" opacity="0.7">Model Extraction</text>
    
    <!-- Arrow 4 - Poisoning -->
    <line x1="1050" y1="420" x2="670" y2="360" stroke="#ff6b6b" stroke-width="2" opacity="0.6" marker-end="url(#red-arrow)"/>
    <text x="950" y="440" font-family="monospace" font-size="8" fill="#ff6b6b" opacity="0.7">Data Poisoning</text>
  </g>
  <!-- Arrow defs -->
  <defs>
    <marker id="red-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#ff6b6b"/>
    </marker>
    <marker id="yellow-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#ffd93d"/>
    </marker>
    <marker id="purple-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="#a78bfa"/>
    </marker>
  </defs>
  <!-- Shield overlay -->
  <g filter="url(#glow)">
    <path d="M 600 240 L 700 275 L 700 350 Q 700 410 600 440 Q 500 410 500 350 L 500 275 Z" fill="none" stroke="#6bcf7f" stroke-width="2" opacity="0.5"/>
    <text x="600" y="355" text-anchor="middle" font-family="monospace" font-size="8" fill="#6bcf7f" opacity="0.6">RED TEAM</text>
  </g>''',
    "#6bcf7f"
)

# 7 — Model Watermarking: Fingerprint on neural net
watermark_svg = wrap(
    "Model Watermarking:",
    "Proving AI Ownership",
    "THEFT DETECTION · IP PROTECTION",
    '''  <!-- Neural network silhouette (faded) -->
  <g filter="url(#glow)" opacity="0.15">
    <circle cx="300" cy="250" r="6" fill="#a78bfa"/>
    <circle cx="300" cy="340" r="6" fill="#a78bfa"/>
    <circle cx="300" cy="430" r="6" fill="#a78bfa"/>
    <circle cx="500" cy="220" r="6" fill="#a78bfa"/>
    <circle cx="500" cy="320" r="6" fill="#a78bfa"/>
    <circle cx="500" cy="420" r="6" fill="#a78bfa"/>
    <circle cx="700" cy="250" r="6" fill="#a78bfa"/>
    <circle cx="700" cy="340" r="6" fill="#a78bfa"/>
    <circle cx="700" cy="430" r="6" fill="#a78bfa"/>
    <circle cx="900" cy="290" r="6" fill="#a78bfa"/>
    <circle cx="900" cy="380" r="6" fill="#a78bfa"/>
    <g stroke="#a78bfa" stroke-width="0.5" opacity="0.2">
      <line x1="306" y1="250" x2="494" y2="220"/><line x1="306" y1="250" x2="494" y2="320"/>
      <line x1="306" y1="340" x2="494" y2="320"/><line x1="306" y1="340" x2="494" y2="420"/>
      <line x1="306" y1="430" x2="494" y2="320"/><line x1="306" y1="430" x2="494" y2="420"/>
      <line x1="506" y1="220" x2="694" y2="250"/><line x1="506" y1="220" x2="694" y2="340"/>
      <line x1="506" y1="320" x2="694" y2="250"/><line x1="506" y1="320" x2="694" y2="340"/>
      <line x1="506" y1="320" x2="694" y2="430"/><line x1="506" y1="420" x2="694" y2="340"/>
      <line x1="506" y1="420" x2="694" y2="430"/>
      <line x1="706" y1="250" x2="894" y2="290"/><line x1="706" y1="340" x2="894" y2="290"/>
      <line x1="706" y1="340" x2="894" y2="380"/><line x1="706" y1="430" x2="894" y2="380"/>
    </g>
  </g>
  <!-- Fingerprint overlay -->
  <g filter="url(#glow)">
    <!-- Fingerprint ridges -->
    <path d="M 560 260 Q 580 250 600 260 Q 620 270 640 260" fill="none" stroke="#00d2ff" stroke-width="2.5" opacity="0.7"/>
    <path d="M 550 290 Q 580 280 600 290 Q 620 300 650 290" fill="none" stroke="#00d2ff" stroke-width="2.5" opacity="0.6"/>
    <path d="M 540 320 Q 570 310 600 320 Q 630 330 660 320" fill="none" stroke="#00d2ff" stroke-width="2.5" opacity="0.5"/>
    <path d="M 550 350 Q 580 340 600 350 Q 620 360 650 350" fill="none" stroke="#00d2ff" stroke-width="2.5" opacity="0.4"/>
    <path d="M 570 380 Q 590 370 600 380 Q 610 390 630 380" fill="none" stroke="#00d2ff" stroke-width="2.5" opacity="0.3"/>
    <!-- Fingerprint arch -->
    <path d="M 570 250 Q 600 230 630 250" fill="none" stroke="#00d2ff" stroke-width="2" opacity="0.5"/>
    <path d="M 555 270 Q 600 245 645 270" fill="none" stroke="#00d2ff" stroke-width="2" opacity="0.5"/>
  </g>
  <!-- Checkmark (ownership verified) -->
  <g filter="url(#glow)">
    <circle cx="850" cy="240" r="25" fill="none" stroke="#6bcf7f" stroke-width="2" opacity="0.6"/>
    <polyline points="838,240 846,250 862,230" fill="none" stroke="#6bcf7f" stroke-width="2.5" opacity="0.8"/>
    <text x="850" y="290" text-anchor="middle" font-family="monospace" font-size="8" fill="#6bcf7f" opacity="0.6">VERIFIED</text>
  </g>''',
    "#00d2ff"
)

# 8 — Eval Benchmark Poisoning: Leaderboard + contamination
eval_svg = wrap(
    "Eval Benchmark Poisoning:",
    "Gaming the Leaderboards",
    "DATA CONTAMINATION · BENCHMARK INTEGRITY",
    '''  <!-- Podium / leaderboard -->
  <g filter="url(#glow)">
    <!-- 1st place -->
    <rect x="520" y="260" width="60" height="100" rx="4" fill="none" stroke="#ffd93d" stroke-width="2" opacity="0.5"/>
    <rect x="525" y="265" width="50" height="90" rx="2" fill="#ffd93d" opacity="0.08"/>
    <text x="550" y="320" text-anchor="middle" font-family="monospace" font-size="10" fill="#ffd93d" opacity="0.8">#1</text>
    
    <!-- 2nd place -->
    <rect x="440" y="290" width="50" height="70" rx="4" fill="none" stroke="#8b949e" stroke-width="1.5" opacity="0.4"/>
    <rect x="445" y="295" width="40" height="60" rx="2" fill="#8b949e" opacity="0.06"/>
    <text x="465" y="330" text-anchor="middle" font-family="monospace" font-size="9" fill="#8b949e" opacity="0.6">#2</text>
    
    <!-- 3rd place -->
    <rect x="610" y="310" width="50" height="50" rx="4" fill="none" stroke="#b87333" stroke-width="1.5" opacity="0.4"/>
    <text x="635" y="340" text-anchor="middle" font-family="monospace" font-size="9" fill="#b87333" opacity="0.6">#3</text>
    
    <!-- Leaderboard labels -->
    <text x="550" y="250" text-anchor="middle" font-family="monospace" font-size="7" fill="#ffd93d" opacity="0.5">"GPT-5"</text>
    <text x="465" y="280" text-anchor="middle" font-family="monospace" font-size="7" fill="#8b949e" opacity="0.4">"Claude 4"</text>
    <text x="635" y="302" text-anchor="middle" font-family="monospace" font-size="7" fill="#b87333" opacity="0.4">"Llama 4"</text>
  </g>
  <!-- Contamination drip from above -->
  <g filter="url(#glow)">
    <!-- Poison drop falling onto #1 -->
    <path d="M 550 180 Q 545 200 550 210 Q 555 200 550 180" fill="#ff6b6b" opacity="0.7"/>
    <path d="M 550 210 L 548 220 Q 548 225 550 225 Q 552 225 552 220 L 550 210" fill="#ff6b6b" opacity="0.5"/>
    <!-- Contamination spread -->
    <ellipse cx="550" cy="240" rx="15" ry="5" fill="#ff6b6b" opacity="0.15"/>
    <ellipse cx="550" cy="235" rx="25" ry="8" fill="#ff6b6b" opacity="0.08"/>
    
    <!-- Cross-contamination arrows -->
    <path d="M 575 235 Q 590 240 610 230" fill="none" stroke="#ff6b6b" stroke-width="1" opacity="0.3" stroke-dasharray="4,3"/>
    <path d="M 525 235 Q 510 240 490 230" fill="none" stroke="#ff6b6b" stroke-width="1" opacity="0.3" stroke-dasharray="4,3"/>
    
    <!-- Contamination label -->
    <text x="550" y="175" text-anchor="middle" font-family="monospace" font-size="8" fill="#ff6b6b" opacity="0.7">CONTAMINATION</text>
  </g>
  <!-- Benchmark name -->
  <g filter="url(#glow)" opacity="0.3">
    <text x="550" y="400" text-anchor="middle" font-family="monospace" font-size="9" fill="#8b949e">MMLU: 89.7%  GSM8K: 92.3%  HumanEval: 91.1%</text>
    <text x="550" y="420" text-anchor="middle" font-family="monospace" font-size="7" fill="#8b949e">↑ All contaminated by training data leakage</text>
  </g>''',
    "#ffd93d"
)

# 9 — Multimodal Attacks: Audio + Video + Text linked
multimodal_svg = wrap(
    "Real-Time Multimodal",
    "Security: Audio/Video Attacks",
    "VOICE INJECTION · VIDEO PROMPTS · OMNIMODAL",
    '''  <!-- Audio waveform (left) -->
  <g filter="url(#glow)">
    <path d="M 150 340 Q 180 280 200 340 Q 220 400 240 340 Q 260 280 280 340 Q 300 400 320 340" fill="none" stroke="#00d2ff" stroke-width="2.5" opacity="0.6"/>
    <path d="M 170 340 Q 190 310 210 340 Q 230 370 250 340 Q 270 310 290 340" fill="none" stroke="#00d2ff" stroke-width="1" opacity="0.3"/>
    <text x="235" y="380" text-anchor="middle" font-family="monospace" font-size="8" fill="#00d2ff" opacity="0.6">AUDIO</text>
  </g>
  <!-- Video play button (center) -->
  <g filter="url(#glow)">
    <rect x="450" y="270" width="100" height="80" rx="6" fill="none" stroke="#6bcf7f" stroke-width="2" opacity="0.5"/>
    <polygon points="480,290 480,330 510,310" fill="#6bcf7f" opacity="0.5"/>
    <text x="500" y="380" text-anchor="middle" font-family="monospace" font-size="8" fill="#6bcf7f" opacity="0.6">VIDEO</text>
    <!-- Video frame content suggestion -->
    <rect x="460" y="278" width="80" height="15" rx="2" fill="#6bcf7f" opacity="0.08"/>
    <line x1="465" y1="283" x2="535" y2="283" stroke="#6bcf7f" stroke-width="0.5" opacity="0.2"/>
    <line x1="465" y1="288" x2="520" y2="288" stroke="#6bcf7f" stroke-width="0.5" opacity="0.2"/>
  </g>
  <!-- Text characters (right) -->
  <g filter="url(#glow)">
    <text x="800" y="290" font-family="monospace" font-size="28" fill="#a78bfa" opacity="0.4">T</text>
    <text x="830" y="290" font-family="monospace" font-size="28" fill="#a78bfa" opacity="0.3">E</text>
    <text x="860" y="290" font-family="monospace" font-size="28" fill="#a78bfa" opacity="0.4">X</text>
    <text x="890" y="290" font-family="monospace" font-size="28" fill="#a78bfa" opacity="0.3">T</text>
    <text x="800" y="340" font-family="monospace" font-size="12" fill="#a78bfa" opacity="0.3">"invisible</text>
    <text x="800" y="360" font-family="monospace" font-size="12" fill="#a78bfa" opacity="0.3"> overrides"</text>
  </g>
  <!-- Attack chain linking them -->
  <g filter="url(#glow)">
    <path d="M 320 330 Q 380 310 440 310" fill="none" stroke="#ff6b6b" stroke-width="2" opacity="0.6" stroke-dasharray="6,4"/>
    <path d="M 560 310 Q 640 310 700 340 Q 740 350 790 320" fill="none" stroke="#ff6b6b" stroke-width="2" opacity="0.6" stroke-dasharray="6,4"/>
    <!-- Attack label -->
    <text x="600" y="240" text-anchor="middle" font-family="monospace" font-size="9" fill="#ff6b6b" opacity="0.7">CROSS-MODAL ATTACK CHAIN</text>
  </g>
  <!-- Hidden injection indicators -->
  <g filter="url(#glow)" opacity="0.4">
    <text x="235" y="400" font-family="monospace" font-size="7" fill="#ff6b6b">🔇 ultrasonic command</text>
    <text x="500" y="400" font-family="monospace" font-size="7" fill="#ff6b6b">📷 frame text overlay</text>
    <text x="870" y="390" font-family="monospace" font-size="7" fill="#ff6b6b">🔤 prompt in plain sight</text>
  </g>''',
    "#00d2ff"
)

# 10 — ML Secrets Management: Key + vault + circuit
secrets_svg = wrap(
    "ML Pipeline Secrets",
    "Management",
    "VAULT · API KEYS · CREDENTIAL HYGIENE",
    '''  <!-- Vault door (center) -->
  <g filter="url(#glow)">
    <rect x="480" y="240" width="140" height="130" rx="8" fill="none" stroke="#00d2ff" stroke-width="2" opacity="0.4"/>
    <rect x="490" y="250" width="120" height="110" rx="4" fill="#00d2ff" opacity="0.04"/>
    <!-- Vault handle -->
    <circle cx="550" cy="320" r="18" fill="none" stroke="#00d2ff" stroke-width="2" opacity="0.5"/>
    <circle cx="550" cy="320" r="6" fill="#00d2ff" opacity="0.3"/>
    <line x1="550" y1="302" x2="550" y2="290" stroke="#00d2ff" stroke-width="2" opacity="0.4"/>
    <!-- Hinges -->
    <rect x="615" y="260" width="5" height="15" rx="2" fill="#00d2ff" opacity="0.3"/>
    <rect x="615" y="345" width="5" height="15" rx="2" fill="#00d2ff" opacity="0.3"/>
  </g>
  <!-- Key -->
  <g filter="url(#glow)">
    <!-- Key shaft -->
    <rect x="300" y="315" width="140" height="8" rx="2" fill="#ffd93d" opacity="0.6"/>
    <!-- Key bow (handle) -->
    <circle cx="295" cy="319" r="20" fill="none" stroke="#ffd93d" stroke-width="2.5" opacity="0.6"/>
    <circle cx="295" cy="319" r="12" fill="none" stroke="#ffd93d" stroke-width="1.5" opacity="0.4"/>
    <!-- Key teeth -->
    <rect x="430" y="315" width="10" height="18" rx="1" fill="#ffd93d" opacity="0.5"/>
    <rect x="415" y="315" width="10" height="14" rx="1" fill="#ffd93d" opacity="0.5"/>
    <rect x="445" y="315" width="8" height="12" rx="1" fill="#ffd93d" opacity="0.4"/>
  </g>
  <!-- Circuit traces connecting key to vault -->
  <g filter="url(#glow)" stroke="#ffd93d" stroke-width="1.5" opacity="0.3" fill="none">
    <path d="M 445 319 L 470 319 L 480 300"/>
    <path d="M 445 322 L 470 322 L 480 350"/>
  </g>
  <!-- API key string -->
  <g filter="url(#glow)" opacity="0.3">
    <text x="550" y="220" text-anchor="middle" font-family="monospace" font-size="8" fill="#ffd93d">OPENAI_API_KEY=sk-********************</text>
    <text x="550" y="410" text-anchor="middle" font-family="monospace" font-size="8" fill="#00d2ff">HF_TOKEN=hf_**************************</text>
  </g>
  <!-- Small lock icons -->
  <g filter="url(#glow)">
    <rect x="780" y="270" width="35" height="28" rx="4" fill="none" stroke="#6bcf7f" stroke-width="1.5" opacity="0.5"/>
    <path d="M 790 270 L 790 260 Q 797 250 805 260 L 805 270" fill="none" stroke="#6bcf7f" stroke-width="1.5" opacity="0.4"/>
    <circle cx="797" cy="282" r="3" fill="#6bcf7f" opacity="0.4"/>
    <line x1="797" y1="285" x2="797" y2="290" stroke="#6bcf7f" stroke-width="1" opacity="0.3"/>
    
    <rect x="780" y="340" width="35" height="28" rx="4" fill="none" stroke="#6bcf7f" stroke-width="1.5" opacity="0.5"/>
    <path d="M 790 340 L 790 330 Q 797 320 805 330 L 805 340" fill="none" stroke="#6bcf7f" stroke-width="1.5" opacity="0.4"/>
    <circle cx="797" cy="352" r="3" fill="#6bcf7f" opacity="0.4"/>
    <line x1="797" y1="355" x2="797" y2="360" stroke="#6bcf7f" stroke-width="1" opacity="0.3"/>
  </g>
  <!-- Credential tokens flowing -->
  <g filter="url(#glow)" opacity="0.15">
    <text x="870" y="290" font-family="monospace" font-size="7" fill="#6bcf7f">AWS_KEY ✓</text>
    <text x="870" y="360" font-family="monospace" font-size="7" fill="#6bcf7f">VAULT_OK ✓</text>
  </g>''',
    "#00d2ff"
)

# Write all SVGs
covers = {
    "cover-memory-attack-surface": memory_svg,
    "cover-automated-science-guardrails": science_svg,
    "cover-multi-agent-collusion": collusion_svg,
    "cover-structured-data-injection": structured_svg,
    "cover-automated-red-teaming": redteam_svg,
    "cover-model-watermarking": watermark_svg,
    "cover-eval-benchmark-poisoning": eval_svg,
    "cover-multimodal-attacks": multimodal_svg,
    "cover-ml-secrets-management": secrets_svg,
}

for slug, svg_content in covers.items():
    if svg_content:
        path = os.path.join(SVG_DIR, f"{slug}.svg")
        with open(path, "w") as f:
            f.write(svg_content)
        print(f"  {slug}.svg — {len(svg_content)} bytes")
