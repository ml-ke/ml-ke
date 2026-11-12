#!/usr/bin/env python3
"""AI Crime Watch series cover generator.

Usage:
  python3 issue_cover.py <slug> <issue_no> <headline> <date> [subtitle] [repo_dir]

Writes assets/blog/cover-<slug>.svg (source) and assets/img/cover-<slug>.webp
(1200x630) inside repo_dir (default: cwd). Requires ffmpeg for the WebP step.
"""
import html
import pathlib
import subprocess
import sys

BG1 = "#0d1117"
BG2 = "#1a1a2e"
RED = "#ff6b6b"
GREEN = "#6bcf7f"
CYAN = "#00d2ff"
YELLOW = "#ffd93d"
GRAY = "#8b949e"
LIGHT = "#e6edf3"


def esc(s: str) -> str:
    return html.escape(s, quote=True)


def split_lines(text: str, width: int = 42) -> list:
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines[:2]  # max two lines; truncate gracefully


def main() -> None:
    args = sys.argv[1:]
    if len(args) < 4:
        print(
            "usage: issue_cover.py <slug> <issue_no> <headline> <date> "
            "[subtitle] [repo_dir]"
        )
        sys.exit(2)
    slug, issue = args[0], args[1]
    headline, date_str = args[2], args[3]
    subtitle = args[4] if len(args) > 4 else ""
    repo = pathlib.Path(args[5]) if len(args) > 5 else pathlib.Path.cwd()
    blog_dir = repo / "assets" / "blog"
    img_dir = repo / "assets" / "img"
    blog_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)

    h_lines = split_lines(headline)
    ys = [380, 424]
    text_y = [f'<text x="600" y="{ys[i]}" text-anchor="middle" '
              f'font-family="DejaVu Sans, Arial, sans-serif" font-size="40" '
              f'font-weight="bold" fill="{LIGHT}">{esc(l)}</text>'
             for i, l in enumerate(h_lines)]
    sub_y = 424 + 22 * max(0, len(h_lines) - 1) + 34
    if subtitle:
        text_y.append(f'<text x="600" y="{sub_y}" text-anchor="middle" '
                      f'font-family="DejaVu Sans, Arial, sans-serif" font-size="22" '
                      f'fill="{CYAN}">{esc(subtitle)}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <linearGradient id="bg" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="{BG1}"/><stop offset="1" stop-color="{BG2}"/>
    </linearGradient>
    <linearGradient id="chipg" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#2d333b"/><stop offset="1" stop-color="#161b22"/>
    </linearGradient>
  </defs>
  <rect width="1200" height="630" fill="url(#bg)"/>
  <g stroke="#ffffff" stroke-opacity="0.03">
    {''.join(f'<line x1="0" y1="{y}" x2="1200" y2="{y}"/>' for y in range(0, 630, 45))}
    {''.join(f'<line x1="{x}" y1="0" x2="{x}" y2="630"/>' for x in range(0, 1200, 45))}
  </g>

  <!-- brand -->
  <text x="46" y="56" font-family="DejaVu Sans, Arial, sans-serif" font-size="24"
        font-weight="bold" letter-spacing="4" fill="{LIGHT}">AI CRIME WATCH</text>
  <text x="46" y="80" font-family="DejaVu Sans, Arial, sans-serif" font-size="13"
        letter-spacing="2" fill="{GRAY}">A FORTNIGHTLY COLLATION ON CRIME &amp; AI</text>
  <!-- issue badge -->
  <rect x="1006" y="34" width="148" height="40" rx="8" fill="{YELLOW}" fill-opacity="0.12"/>
  <text x="1080" y="60" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif"
        font-size="18" font-weight="bold" fill="{YELLOW}">ISSUE {esc(issue)}</text>

  <!-- scales of justice: beam + fulcrum -->
  <rect x="320" y="196" width="560" height="10" rx="5" fill="{LIGHT}"/>
  <polygon points="588,206 612,206 600,268" fill="{LIGHT}"/>
  <line x1="600" y1="268" x2="600" y2="340" stroke="{GRAY}" stroke-width="4"/>
  <circle cx="600" cy="344" r="10" fill="{GRAY}"/>
  <!-- left chains + pan (crime side: AI chip) -->
  <line x1="410" y1="206" x2="410" y2="280" stroke="{RED}" stroke-width="4"/>
  <line x1="490" y1="206" x2="490" y2="280" stroke="{RED}" stroke-width="4"/>
  <path d="M360 300 Q360 342 410 342 Q460 342 460 300 Z" fill="none" stroke="{RED}" stroke-width="5"/>
  <path d="M360 300 Q410 306 460 300 L460 322 Q410 334 360 322 Z" fill="{RED}" fill-opacity="0.15"/>
  <!-- chip on left pan -->
  <rect x="390" y="262" width="40" height="40" rx="4" fill="url(#chipg)" stroke="{RED}" stroke-width="2"/>
  {''.join(f'<line x1="{x}" y1="258" x2="{x}" y2="306" stroke="{RED}" stroke-width="2"/>' for x in range(396, 426, 8))}
  {''.join(f'<line x1="386" y1="{y}" x2="434" y2="{y}" stroke="{RED}" stroke-width="2"/>' for y in range(266, 300, 8))}
  <rect x="404" y="276" width="12" height="12" rx="2" fill="{RED}" fill-opacity="0.5"/>
  <text x="410" y="360" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif"
        font-size="15" fill="{RED}">CRIME</text>
  <!-- right chains + pan (justice side: shield) -->
  <line x1="710" y1="206" x2="710" y2="280" stroke="{GREEN}" stroke-width="4"/>
  <line x1="790" y1="206" x2="790" y2="280" stroke="{GREEN}" stroke-width="4"/>
  <path d="M660 300 Q660 342 710 342 Q760 342 760 300 Z" fill="none" stroke="{GREEN}" stroke-width="5"/>
  <path d="M660 300 Q710 306 760 300 L760 322 Q710 334 660 322 Z" fill="{GREEN}" fill-opacity="0.15"/>
  <!-- shield on right pan -->
  <path d="M710 252 L742 268 L742 294 C742 312 728 324 710 330 C692 324 678 312 678 294 L678 268 Z"
        fill="{GREEN}" fill-opacity="0.16" stroke="{GREEN}" stroke-width="4"/>
  <path d="M710 268 L732 279 L732 294 C732 306 723 314 710 319 C697 314 688 306 688 294 L688 279 Z"
        fill="none" stroke="{LIGHT}" stroke-width="2"/>
  <text x="710" y="360" text-anchor="middle" font-family="DejaVu Sans, Arial, sans-serif"
        font-size="15" fill="{GREEN}">JUSTICE</text>

  <!-- headline block -->
  {''.join(text_y)}

  <text x="46" y="598" font-family="DejaVu Sans, Arial, sans-serif" font-size="15" fill="{GRAY}">{esc(date_str)}</text>
  <text x="1154" y="598" text-anchor="end" font-family="DejaVu Sans, Arial, sans-serif"
        font-size="15" fill="{GRAY}">ml-ke.github.io</text>
</svg>
'''

    svg_path = blog_dir / f"cover-{slug}.svg"
    svg_path.write_text(svg)
    webp_path = img_dir / f"cover-{slug}.webp"
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error", "-i", str(svg_path),
        "-vf", "scale=1200:630", "-c:v", "libwebp", "-quality", "80",
        str(webp_path),
    ]
    subprocess.run(cmd, check=True)
    print(f"Wrote {svg_path}")
    print(f"Wrote {webp_path} ({webp_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
