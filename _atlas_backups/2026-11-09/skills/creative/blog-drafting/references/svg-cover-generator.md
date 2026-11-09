# SVG Cover Image Generator

## Quick Start
```bash
python3 tools/generate_covers.py
```
This recreates all cover SVGs in `assets/blog/`.

## Generator Script
Location: `tools/generate_covers.py`

## Supported Visual Styles
| Style Key | Description | Used For |
|-----------|-------------|----------|
| `graph` | Nodes connected by edges, central cluster | KG/neo4j posts |
| `vectors` | Grid of squares labeled h₁, h₂, h₃... | Embedding posts |
| `neural` | Multi-layer neural network with connections | GNN posts |
| `infra` | Boxes labeled DB, CACHE, API, INDEX | Production/infra posts |
| `hybrid` | Graph half + LLM half separated by dashed line | KG+LLM posts |
| `danger` | Warning triangle with exclamation mark | Security risk posts |
| `exploit` | User input → LLM → Output flow diagram | Jailbreak posts |
| `poison` | Dataset with needle + trained model with trigger | Data poisoning posts |
| `agent` | Agent connecting to multiple tools, alert icon | Agent security posts |
| `chain` | Connected boxes (REPO → CI/CD → DEPLOY → MODEL) | Supply chain posts |

## Color Themes
Each style maps to a primary/secondary/accent color triple. The color determines the node outlines, text highlights, and glow effects.

## SVG Structure
- 1200×620 viewport
- Dark linear gradient background (#0d1117 → #161b22 → #1a1a2e)
- Grid opacity lines (opacity 0.03)
- Decorative elements with `filter="url(#glow)"` for glow effect
- Title at y=140 and y=182, centered, 34px bold
- Subtitle at y=520 with letter-spacing: 3
- Decorative line at y=535
- Branding at bottom-right: "ml-ke.github.io" (y=600)
