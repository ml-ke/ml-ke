#!/usr/bin/env python3
"""Generate 3 unique cover SVGs for the staged posts (Aug 24, 26, 27)."""

def base(title, subtitle, accent, elements, title_size=33):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#0d1117"/>
      <stop offset="1" stop-color="#1a1a2e"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="630" fill="url(#bg)"/>
  <g stroke="#ffffff" stroke-opacity="0.03" stroke-width="1">
    <line x1="0" y1="105" x2="1200" y2="105"/><line x1="0" y1="210" x2="1200" y2="210"/>
    <line x1="0" y1="315" x2="1200" y2="315"/><line x1="0" y1="420" x2="1200" y2="420"/>
    <line x1="0" y1="525" x2="1200" y2="525"/>
    <line x1="200" y1="0" x2="200" y2="630"/><line x1="400" y1="0" x2="400" y2="630"/>
    <line x1="600" y1="0" x2="600" y2="630"/><line x1="800" y1="0" x2="800" y2="630"/>
    <line x1="1000" y1="0" x2="1000" y2="630"/>
  </g>
  <text x="600" y="62" text-anchor="middle" font-family="sans-serif" font-size="{title_size}" font-weight="bold" fill="#f0f6fc">{title}</text>
  <text x="600" y="96" text-anchor="middle" font-family="sans-serif" font-size="20" fill="#8b949e">{subtitle}</text>
  <g transform="translate(600, 360)">
    {elements}
  </g>
  <text x="1140" y="600" text-anchor="end" font-family="sans-serif" font-size="18" fill="#6bcf7f">ml-ke.github.io</text>
</svg>
'''

covers = {}

# Aug 24 — KYC/AML analytics: ID document + fingerprint + risk gauge
covers["cover-kyc-aml-analytics-african-fintech.svg"] = base(
    "KYC/AML Analytics for African Fintech",
    "Identity verification, transaction monitoring &amp; data minimization",
    "#00d2ff",
    '''
    <g transform="translate(-150, 0)">
      <rect x="-58" y="-75" width="116" height="150" rx="8" fill="none" stroke="#f0f6fc" stroke-width="3"/>
      <circle cx="0" cy="-28" r="20" fill="none" stroke="#8b949e" stroke-width="3"/>
      <circle cx="0" cy="-28" r="8" fill="#8b949e" opacity="0.5"/>
      <rect x="-40" y="8" width="80" height="10" rx="4" fill="#8b949e" opacity="0.6"/>
      <rect x="-40" y="28" width="60" height="10" rx="4" fill="#8b949e" opacity="0.4"/>
      <rect x="-40" y="48" width="70" height="10" rx="4" fill="#8b949e" opacity="0.4"/>
    </g>
    <g transform="translate(60, 0)">
      <circle r="60" fill="none" stroke="#a78bfa" stroke-width="3" opacity="0.6"/>
      <circle r="42" fill="none" stroke="#a78bfa" stroke-width="3" opacity="0.6"/>
      <path d="M0,52 C-19,36 -24,6 -12,-16 C-2,-34 30,-38 42,-18 C52,-2 34,24 16,36 C9,42 3,46 0,52 Z" fill="none" stroke="#f0f6fc" stroke-width="4"/>
      <path d="M0,52 C-10,18 -5,-18 12,-28" fill="none" stroke="#ff6b6b" stroke-width="2.5" opacity="0.9"/>
    </g>
    <g transform="translate(210, 20)">
      <path d="M-40,40 A40,40 0 1 1 40,40 Z" fill="none" stroke="#6bcf7f" stroke-width="5"/>
      <line x1="0" y1="40" x2="0" y2="-10" stroke="#8b949e" stroke-width="2"/>
      <circle cx="0" cy="-30" r="7" fill="#ffd93d"/>
      <line x1="0" y1="-30" x2="24" y2="18" stroke="#ffd93d" stroke-width="3" stroke-linecap="round"/>
    </g>
    <text x="0" y="150" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#8b949e">analytics for trust, not surveillance</text>
    '''
)

# Aug 26 — LLM security: chat bubble + red injection needle
covers["cover-llm-security-financial-chatbots.svg"] = base(
    "LLM Security for Financial Chatbots",
    "Prompt injection, excessive agency &amp; the OWASP LLM Top 10",
    "#ff6b6b",
    '''
    <g transform="translate(-120, -20)">
      <rect x="-90" y="-60" width="180" height="110" rx="22" fill="none" stroke="#00d2ff" stroke-width="4"/>
      <text x="-50" y="-10" font-family="sans-serif" font-size="17" fill="#f0f6fc">transfer 500 to</text>
      <text x="-50" y="16" font-family="sans-serif" font-size="17" fill="#f0f6fc">account 12345?</text>
      <polygon points="-60,50 -80,84 -40,50" fill="#0d1117"/>
      <polygon points="-60,50 -76,78 -44,50" fill="none" stroke="#00d2ff" stroke-width="3"/>
    </g>
    <g transform="rotate(-24, 60, 0)">
      <rect x="48" y="-130" width="24" height="66" rx="12" fill="#ff6b6b"/>
      <polygon points="60,-130 36,-108 84,-108" fill="#ff6b6b"/>
      <line x1="10" y1="-210" x2="110" y2="-210" stroke="#ff6b6b" stroke-width="10" stroke-linecap="round"/>
      <rect x="30" y="-160" width="60" height="22" rx="4" fill="#a78bfa"/>
    </g>
    <g transform="translate(190, 40)">
      <circle cx="0" cy="0" r="7" fill="#ffd93d"/><circle cx="40" cy="-20" r="5" fill="#ffd93d"/>
      <circle cx="30" cy="25" r="5" fill="#ffd93d"/><circle cx="70" cy="0" r="5" fill="#ffd93d"/>
      <line x1="7" y1="0" x2="33" y2="-17" stroke="#ffd93d" stroke-width="2"/>
      <line x1="7" y1="0" x2="26" y2="22" stroke="#ffd93d" stroke-width="2"/>
      <line x1="35" y1="-15" x2="66" y2="-4" stroke="#ffd93d" stroke-width="2"/>
      <line x1="35" y1="20" x2="65" y2="4" stroke="#ffd93d" stroke-width="2"/>
    </g>
    <text x="0" y="165" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#8b949e">never let the model touch the money directly</text>
    '''
)

# Aug 27 — MLOps/RegTech: pipeline stages + scales + checklist
covers["cover-mlops-regtech-model-governance.svg"] = base(
    "MLOps for RegTech: Model Governance",
    "CBK, ODPC &amp; EU AI Act — the regulated ML lifecycle",
    "#a78bfa",
    '''
    <g stroke="#8b949e" stroke-width="3">
      <line x1="-170" y1="-70" x2="170" y2="-70"/>
    </g>
    <g>
      <circle cx="-150" cy="-70" r="17" fill="#0d1117" stroke="#00d2ff" stroke-width="4"/>
      <text x="-150" y="-65" text-anchor="middle" font-family="monospace" font-size="15" fill="#00d2ff">1</text>
      <circle cx="-90" cy="-70" r="17" fill="#0d1117" stroke="#00d2ff" stroke-width="4"/>
      <text x="-90" y="-65" text-anchor="middle" font-family="monospace" font-size="15" fill="#00d2ff">2</text>
      <circle cx="-30" cy="-70" r="17" fill="#0d1117" stroke="#ffd93d" stroke-width="4"/>
      <text x="-30" y="-65" text-anchor="middle" font-family="monospace" font-size="15" fill="#ffd93d">3</text>
      <circle cx="30" cy="-70" r="17" fill="#0d1117" stroke="#ffd93d" stroke-width="4"/>
      <text x="30" y="-65" text-anchor="middle" font-family="monospace" font-size="15" fill="#ffd93d">4</text>
      <circle cx="90" cy="-70" r="17" fill="#0d1117" stroke="#6bcf7f" stroke-width="4"/>
      <text x="90" y="-65" text-anchor="middle" font-family="monospace" font-size="15" fill="#6bcf7f">5</text>
      <circle cx="150" cy="-70" r="17" fill="#0d1117" stroke="#6bcf7f" stroke-width="4"/>
      <text x="150" y="-65" text-anchor="middle" font-family="monospace" font-size="15" fill="#6bcf7f">6</text>
    </g>
    <g transform="translate(-140, 70)">
      <g stroke="#f0f6fc" stroke-width="3">
        <line x1="0" y1="-40" x2="0" y2="40"/>
        <line x1="-14" y1="-40" x2="14" y2="-40"/>
        <line x1="-14" y1="40" x2="14" y2="40"/>
        <line x1="14" y1="-40" x2="24" y2="-24"/>
        <line x1="24" y1="-24" x2="24" y2="24"/>
        <line x1="24" y1="24" x2="14" y2="40"/>
        <line x1="0" y1="-22" x2="0" y2="22"/>
        <line x1="-8" y1="-10" x2="8" y2="-10"/>
        <line x1="-8" y1="10" x2="8" y2="10"/>
      </g>
      <text x="0" y="70" text-anchor="middle" font-family="monospace" font-size="12" fill="#8b949e">fair?</text>
    </g>
    <g transform="translate(150, 60)">
      <rect x="-58" y="-50" width="116" height="100" rx="8" fill="none" stroke="#6bcf7f" stroke-width="3"/>
      <polyline points="-42,-28 -10,-28 -10,-2 20,-2 20,24 44,24" fill="none" stroke="#6bcf7f" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
      <text x="0" y="70" text-anchor="middle" font-family="monospace" font-size="12" fill="#8b949e">monitor</text>
    </g>
    <text x="0" y="170" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#8b949e">every model decision is a regulated decision</text>
    '''
)

import os
os.makedirs("/home/pro-g/ProG/ml-ke/assets/blog", exist_ok=True)
for name, content in covers.items():
    path = f"/home/pro-g/ProG/ml-ke/assets/blog/{name}"
    with open(path, "w") as f:
        f.write(content)
    print("wrote", path)
