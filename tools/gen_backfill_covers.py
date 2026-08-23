#!/usr/bin/env python3
"""Generate 7 unique blog cover SVGs for the Aug 15-21 backfill posts."""
import os

BG_TOP = "#0d1117"
BG_BOT = "#1a1a2e"

def base(title, subtitle, accent, elements, title_size=33):
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="{BG_TOP}"/>
      <stop offset="1" stop-color="{BG_BOT}"/>
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

# Aug 15 — reconciliation analytics: magnifier over ledger rows + bar chart
covers["cover-reconciliation-analytics-fintech.svg"] = base(
    "Reconciliation Analytics: When the Money Doesn\u2019t Match",
    "Ledger-to-settlement matching, control totals &amp; break detection",
    "#6bcf7f",
    '''
    <g stroke="#6bcf7f" stroke-width="4">
      <line x1="-180" y1="60" x2="-180" y2="-60"/><line x1="-180" y1="60" x2="-40" y2="60"/>
    </g>
    <g fill="#6bcf7f" opacity="0.9">
      <rect x="-170" y="10" width="28" height="50" rx="3"/><rect x="-130" y="-20" width="28" height="80" rx="3"/>
      <rect x="-90" y="30" width="28" height="30" rx="3"/><rect x="-50" y="-40" width="28" height="100" rx="3"/>
    </g>
    <g stroke="#ffd93d" stroke-width="3" stroke-dasharray="6 4" opacity="0.8">
      <line x1="-180" y1="-60" x2="180" y2="-60"/>
    </g>
    <text x="-110" y="120" text-anchor="middle" font-family="monospace" font-size="15" fill="#ff6b6b">ledger \u2260 settlement \u2192 BREAK</text>
    <g transform="translate(120, -20)">
      <circle r="70" fill="none" stroke="#00d2ff" stroke-width="5"/>
      <line x1="10" y1="10" x2="70" y2="70" stroke="#00d2ff" stroke-width="5" stroke-linecap="round"/>
      <rect x="-30" y="-45" width="60" height="90" rx="6" fill="none" stroke="#00d2ff" stroke-width="3" transform="rotate(-45 0 0)"/>
    </g>
    <text x="120" y="120" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#8b949e">catch integrity failures, not just uptime</text>
    '''
)

# Aug 16 — insider threat: key + fingerprint + clock
covers["cover-insider-threat-privileged-access-fintech.svg"] = base(
    "The Three-Minute Insider: Privileged Access as the Attack",
    "Least privilege, JIT access, session recording &amp; UEBA",
    "#ff6b6b",
    '''
    <g transform="translate(-150, 10)">
      <circle r="62" fill="none" stroke="#00d2ff" stroke-width="4"/>
      <line x1="0" y1="0" x2="-28" y2="8" stroke="#f0f6fc" stroke-width="7" stroke-linecap="round"/>
      <line x1="0" y1="0" x2="-47" y2="-15" stroke="#ff6b6b" stroke-width="4" stroke-linecap="round"/>
      <circle r="4" fill="#ffd93d"/>
      <text x="0" y="95" text-anchor="middle" font-family="monospace" font-size="15" fill="#ff6b6b">05:33 AM</text>
    </g>
    <g transform="translate(60, 0)">
      <circle r="65" fill="none" stroke="#a78bfa" stroke-width="3" opacity="0.6"/>
      <circle r="48" fill="none" stroke="#a78bfa" stroke-width="3" opacity="0.6"/>
      <path d="M0,58 C-22,40 -28,8 -14,-18 C-2,-40 34,-44 48,-22 C60,-4 40,26 18,40 C10,46 2,52 0,58 Z" fill="none" stroke="#f0f6fc" stroke-width="4"/>
      <path d="M0,58 C-12,20 -6,-22 14,-32" fill="none" stroke="#ff6b6b" stroke-width="3" opacity="0.9"/>
      <circle cx="14" cy="-32" r="6" fill="none" stroke="#ff6b6b" stroke-width="2.5"/>
    </g>
    <g transform="translate(200, 40)">
      <circle r="26" fill="none" stroke="#ffd93d" stroke-width="5"/>
      <path d="M-14,0 L-4,0 L6,-18 L14,-18 L6,8 L-4,8 Z" fill="none" stroke="#ffd93d" stroke-width="4" stroke-linejoin="round"/>
    </g>
    <text x="0" y="150" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#8b949e">trusted access is the attack surface</text>
    '''
)

# Aug 17 — vendor risk: chain links + contract document
covers["cover-vendor-risk-fintech-contractors.svg"] = base(
    "When the Contractor Has the Keys",
    "Vendor due diligence, scope-limited access &amp; lifecycle governance",
    "#ffd93d",
    '''
    <g transform="translate(-140, 0)">
      <rect x="-60" y="-70" width="120" height="150" rx="8" fill="none" stroke="#f0f6fc" stroke-width="3"/>
      <line x1="-40" y1="-40" x2="40" y2="-40" stroke="#8b949e" stroke-width="2"/>
      <line x1="-40" y1="-20" x2="40" y2="-20" stroke="#8b949e" stroke-width="2"/>
      <line x1="-40" y1="0" x2="20" y2="0" stroke="#8b949e" stroke-width="2"/>
      <text x="0" y="52" text-anchor="middle" font-family="monospace" font-size="13" fill="#ffd93d">VENDOR CONTRACT</text>
    </g>
    <g stroke="#00d2ff" stroke-width="7" fill="none">
      <circle cx="-30" cy="-40" r="16"/><circle cx="30" cy="-40" r="16"/>
      <circle cx="-30" cy="40" r="16"/><circle cx="30" cy="40" r="16"/>
    </g>
    <g transform="translate(150, 0)">
      <rect x="-55" y="-75" width="110" height="150" rx="8" fill="none" stroke="#ff6b6b" stroke-width="4" stroke-dasharray="8 6"/>
      <circle cx="0" cy="-30" r="18" fill="none" stroke="#ffd93d" stroke-width="4"/>
      <line x1="0" y1="-48" x2="0" y2="-40" stroke="#ffd93d" stroke-width="4" stroke-linecap="round"/>
      <line x1="-10" y1="-28" x2="10" y2="-28" stroke="#ffd93d" stroke-width="4" stroke-linecap="round"/>
      <text x="0" y="22" text-anchor="middle" font-family="monospace" font-size="13" fill="#ff6b6b">LIVE PROD</text>
      <text x="0" y="44" text-anchor="middle" font-family="monospace" font-size="13" fill="#ff6b6b">ACCESS</text>
    </g>
    <text x="0" y="145" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#8b949e">the maintenance window is a schedule, not a control</text>
    '''
)

# Aug 18 — Tuesday AI roundup: globe + region nodes
covers["cover-tuesday-ai-update.svg"] = base(
    "Global AI Roundup \u2014 August 2026",
    "Seven regions, one month of AI news",
    "#00d2ff",
    '''
    <circle r="130" fill="none" stroke="#00d2ff" stroke-width="3" opacity="0.35"/>
    <ellipse rx="130" ry="46" fill="none" stroke="#00d2ff" stroke-width="2" opacity="0.5"/>
    <line x1="-130" y1="0" x2="130" y2="0" stroke="#00d2ff" stroke-width="2" opacity="0.5"/>
    <ellipse rx="48" ry="130" fill="none" stroke="#00d2ff" stroke-width="1.5" opacity="0.3"/>
    <g fill="#ffd93d">
      <circle cx="-95" cy="-70" r="9"/><circle cx="100" cy="-80" r="9"/><circle cx="20" cy="-110" r="9"/>
      <circle cx="-120" cy="40" r="9"/><circle cx="110" cy="50" r="9"/><circle cx="-30" cy="105" r="9"/>
      <circle cx="70" cy="95" r="9"/>
    </g>
    <g stroke="#6bcf7f" stroke-width="2" opacity="0.8">
      <line x1="-86" y1="-70" x2="20" y2="-110"/><line x1="20" y1="-110" x2="100" y2="-80"/>
      <line x1="-95" y1="-70" x2="-120" y2="40"/><line x1="100" y1="-80" x2="110" y2="50"/>
      <line x1="-120" y1="40" x2="-30" y2="105"/><line x1="110" y1="50" x2="70" y2="95"/>
      <line x1="-30" y1="105" x2="70" y2="95"/>
    </g>
    <text x="0" y="170" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#8b949e">Asia \u00b7 Europe \u00b7 MENA \u00b7 Africa \u00b7 LatAm \u00b7 Russia \u00b7 West</text>
    '''
)

# Aug 19 — AI-enabled security: AI eye + shield + anomaly dots
covers["cover-ai-enabled-security-anomaly-detection.svg"] = base(
    "AI-Enabled Security: Seeing the Anomaly",
    "UEBA, unsupervised detection &amp; graph analytics for fraud",
    "#a78bfa",
    '''
    <g transform="translate(-130, 0)">
      <ellipse rx="75" ry="42" fill="none" stroke="#00d2ff" stroke-width="5"/>
      <circle r="20" fill="#00d2ff"/>
      <circle r="8" fill="#0d1117"/>
    </g>
    <g stroke="#a78bfa" stroke-width="2.5">
      <circle cx="60" cy="-60" r="5" fill="#a78bfa"/><circle cx="110" cy="-30" r="5" fill="#a78bfa"/>
      <circle cx="90" cy="30" r="5" fill="#a78bfa"/><circle cx="40" cy="70" r="5" fill="#a78bfa"/>
      <circle cx="120" cy="70" r="5" fill="#a78bfa"/>
    </g>
    <g fill="#ff6b6b">
      <circle cx="90" cy="-40" r="8"/><circle cx="100" cy="40" r="8"/><circle cx="60" cy="80" r="8"/>
    </g>
    <g stroke="#ff6b6b" stroke-width="3" stroke-dasharray="6 5" opacity="0.8">
      <ellipse cx="90" cy="-40" rx="34" ry="26"/>
      <ellipse cx="100" cy="40" rx="34" ry="26"/>
    </g>
    <g transform="translate(150, 100)">
      <path d="M0,-50 L45,-32 L45,10 C45,40 20,55 0,62 C-20,55 -45,40 -45,10 L-45,-32 Z" fill="none" stroke="#6bcf7f" stroke-width="4"/>
      <polyline points="-14,-6 2,10 22,-12" fill="none" stroke="#6bcf7f" stroke-width="5" stroke-linecap="round" stroke-linejoin="round"/>
    </g>
    <text x="0" y="150" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#8b949e">rules catch known fraud \u00b7 AI catches unknown fraud \u00b7 humans review</text>
    '''
)

# Aug 20 — fraud ML: graph nodes + red cluster + threshold
covers["cover-fraud-ml-mobile-money.svg"] = base(
    "Fraud ML in Mobile Money",
    "Velocity, graph features &amp; the 70-account loophole",
    "#6bcf7f",
    '''
    <g stroke="#8b949e" stroke-width="2" opacity="0.7">
      <line x1="-140" y1="0" x2="-60" y2="-50"/><line x1="-60" y1="-50" x2="20" y2="-40"/>
      <line x1="-140" y1="0" x2="-70" y2="50"/><line x1="-70" y1="50" x2="20" y2="-40"/>
      <line x1="20" y1="-40" x2="110" y2="-70"/><line x1="20" y1="-40" x2="100" y2="30"/>
      <line x1="-70" y1="50" x2="100" y2="30"/><line x1="-70" y1="50" x2="60" y2="90"/>
    </g>
    <g fill="#00d2ff">
      <circle cx="-140" cy="0" r="9"/><circle cx="-60" cy="-50" r="9"/><circle cx="-70" cy="50" r="9"/>
    </g>
    <g fill="#ffd93d">
      <circle cx="20" cy="-40" r="11"/><circle cx="110" cy="-70" r="8"/><circle cx="100" cy="30" r="8"/>
      <circle cx="60" cy="90" r="8"/>
    </g>
    <g fill="#ff6b6b">
      <circle cx="-70" cy="50" r="12"/><circle cx="-140" cy="0" r="12"/>
    </g>
    <g stroke="#ff6b6b" stroke-width="3" stroke-dasharray="6 5" opacity="0.9">
      <ellipse cx="-105" cy="25" rx="55" ry="45"/>
    </g>
    <g transform="translate(-160, 110)">
      <line x1="0" y1="30" x2="280" y2="30" stroke="#8b949e" stroke-width="2"/>
      <line x1="0" y1="30" x2="60" y2="8" stroke="#6bcf7f" stroke-width="3"/>
      <line x1="60" y1="8" x2="150" y2="14" stroke="#6bcf7f" stroke-width="3"/>
      <line x1="150" y1="14" x2="240" y2="-14" stroke="#ff6b6b" stroke-width="3"/>
      <line x1="200" y1="40" x2="200" y2="-20" stroke="#ffd93d" stroke-width="2" stroke-dasharray="5 5"/>
      <text x="200" y="55" text-anchor="middle" font-family="monospace" font-size="12" fill="#ffd93d">threshold</text>
    </g>
    <text x="0" y="170" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#8b949e">the cluster was visible \u2014 nobody was looking at the graph</text>
    '''
)

# Aug 21 — CI/CD: pipeline stages + lock + gate
covers["cover-cicd-security-control-fintech.svg"] = base(
    "CI/CD as a Security Control",
    "Branch protection, approvals &amp; immutable artifacts",
    "#00d2ff",
    '''
    <g stroke="#8b949e" stroke-width="3">
      <line x1="-170" y1="-90" x2="170" y2="-90"/>
    </g>
    <g>
      <circle cx="-150" cy="-90" r="18" fill="#0d1117" stroke="#00d2ff" stroke-width="4"/>
      <text x="-150" y="-85" text-anchor="middle" font-family="monospace" font-size="16" fill="#00d2ff">1</text>
      <circle cx="-90" cy="-90" r="18" fill="#0d1117" stroke="#00d2ff" stroke-width="4"/>
      <text x="-90" y="-85" text-anchor="middle" font-family="monospace" font-size="16" fill="#00d2ff">2</text>
      <circle cx="-30" cy="-90" r="18" fill="#0d1117" stroke="#ffd93d" stroke-width="4"/>
      <text x="-30" y="-85" text-anchor="middle" font-family="monospace" font-size="16" fill="#ffd93d">3</text>
      <circle cx="30" cy="-90" r="18" fill="#0d1117" stroke="#ffd93d" stroke-width="4"/>
      <text x="30" y="-85" text-anchor="middle" font-family="monospace" font-size="16" fill="#ffd93d">4</text>
      <circle cx="90" cy="-90" r="18" fill="#0d1117" stroke="#6bcf7f" stroke-width="4"/>
      <text x="90" y="-85" text-anchor="middle" font-family="monospace" font-size="16" fill="#6bcf7f">5</text>
      <circle cx="150" cy="-90" r="18" fill="#0d1117" stroke="#6bcf7f" stroke-width="4"/>
      <text x="150" y="-85" text-anchor="middle" font-family="monospace" font-size="16" fill="#6bcf7f">6</text>
    </g>
    <g transform="translate(-140, 40)">
      <rect x="-52" y="-40" width="104" height="80" rx="8" fill="none" stroke="#a78bfa" stroke-width="4"/>
      <circle cx="0" cy="-12" r="15" fill="none" stroke="#a78bfa" stroke-width="4"/>
      <path d="M-26,28 L-12,28 L0,6 L12,28 L26,28 Z" fill="none" stroke="#a78bfa" stroke-width="4" stroke-linejoin="round"/>
    </g>
    <g transform="translate(150, 40)">
      <rect x="-52" y="-40" width="104" height="80" rx="8" fill="none" stroke="#ff6b6b" stroke-width="4" stroke-dasharray="8 6"/>
      <text x="0" y="10" text-anchor="middle" font-family="monospace" font-size="15" fill="#ff6b6b">PROD</text>
      <text x="0" y="34" text-anchor="middle" font-family="monospace" font-size="12" fill="#8b949e">approve?</text>
    </g>
    <text x="0" y="165" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#8b949e">a 5:33 AM change can\u2019t reach prod without review</text>
    '''
)

os.makedirs("/home/pro-g/ProG/ml-ke/assets/blog", exist_ok=True)
for name, content in covers.items():
    path = f"/home/pro-g/ProG/ml-ke/assets/blog/{name}"
    with open(path, "w") as f:
        f.write(content)
    print("wrote", path, len(content), "bytes")
