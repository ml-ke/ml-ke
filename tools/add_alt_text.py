"""Re-run to add alt text back to all post frontmatter."""
import os, re

POSTS_DIR = "/home/pro-g/ProG/ml-ke/_posts"
LQIP_DIR = "/home/pro-g/ProG/ml-ke/assets/img/lqip"

# Load all LQIP data URIs
lqip_cache = {}
for fname in os.listdir(LQIP_DIR):
    if fname.endswith('.txt'):
        name = fname[:-4]
        with open(os.path.join(LQIP_DIR, fname)) as f:
            lqip_cache[name] = f.read().strip()

# Front matter mapping with alt text
updates = {
    "2025-11-20-intro-to-gradient-descent.md": {
        "img": "/assets/img/gradient-descent.webp",
        "alt": "Gradient Descent visualization",
        "lqip": "gradient-descent"
    },
    "2025-11-20-momentum-adaptive-learning-rates.md": {
        "img": "/assets/img/momentum-adaptive-lr.webp",
        "alt": "Momentum and Adaptive Learning Rates",
        "lqip": "momentum-adaptive-lr"
    },
    "2025-11-24-second-order-optimization-methods.md": {
        "img": "/assets/img/second-order-optimization.webp",
        "alt": "Second-Order Optimization Methods",
        "lqip": "second-order-optimization"
    },
    "2025-11-24-knowledge-graphs-fundamentals.md": {
        "img": "/assets/img/kg-fundamentals.webp",
        "alt": "Knowledge Graphs Fundamentals",
        "lqip": "kg-fundamentals"
    },
    "2025-12-01-building-knowledge-graph-neo4j-python.md": {
        "img": "/assets/img/neo4j-tutorial.webp",
        "alt": "Building a Knowledge Graph with Neo4j and Python",
        "lqip": "neo4j-tutorial"
    },
    "2025-12-02-graph-algorithms-neo4j.md": {
        "img": "/assets/img/graph-algorithms-neo4j.webp",
        "alt": "Graph Algorithms in Neo4j",
        "lqip": "graph-algorithms-neo4j"
    },
    "2025-12-04-graph-neural-networks-fundamentals.md": {
        "img": "/assets/img/gnn-fundamentals.webp",
        "alt": "Graph Neural Networks Demystified",
        "lqip": "gnn-fundamentals"
    },
    "2026-06-01-kg-embeddings.md": {
        "img": "/assets/img/cover-kg-embeddings.webp",
        "alt": "Knowledge Graph Embeddings from TransE to RotatE",
        "lqip": "cover-kg-embeddings"
    },
    "2026-06-01-gnn-knowledge-graph-reasoning.md": {
        "img": "/assets/img/cover-kg-gnn.webp",
        "alt": "Graph Neural Networks for Knowledge Graph Reasoning",
        "lqip": "cover-kg-gnn"
    },
    "2026-06-01-kg-production.md": {
        "img": "/assets/img/cover-kg-production.webp",
        "alt": "Knowledge Graphs in Production cover",
        "lqip": "cover-kg-production"
    },
    "2026-06-01-kg-llm-rag.md": {
        "img": "/assets/img/cover-kg-llm.webp",
        "alt": "Knowledge Graphs Meet LLMs RAG with Structured Knowledge",
        "lqip": "cover-kg-llm"
    },
    "2026-06-01-prompt-injection-llm-security.md": {
        "img": "/assets/img/cover-ai-prompt-injection.webp",
        "alt": "Prompt Injection The Number One LLM Security Risk",
        "lqip": "cover-ai-prompt-injection"
    },
    "2026-06-01-jailbreaking-llms.md": {
        "img": "/assets/img/cover-ai-jailbreak.webp",
        "alt": "Jailbreaking LLMs from DAN to GODMODE",
        "lqip": "cover-ai-jailbreak"
    },
    "2026-06-01-data-poisoning-model-backdoors.md": {
        "img": "/assets/img/cover-ai-data-poisoning.webp",
        "alt": "Data Poisoning and Model Backdoors",
        "lqip": "cover-ai-data-poisoning"
    },
    "2026-06-01-insecure-agent-design.md": {
        "img": "/assets/img/cover-ai-agent-security.webp",
        "alt": "Insecure Agent Design When AI Has Too Much Agency",
        "lqip": "cover-ai-agent-security"
    },
    "2026-06-01-supply-chain-ai-attacks.md": {
        "img": "/assets/img/cover-ai-supply-chain.webp",
        "alt": "Supply Chain Attacks on AI Systems",
        "lqip": "cover-ai-supply-chain"
    },
}

for fname, vals in updates.items():
    fpath = os.path.join(POSTS_DIR, fname)
    if not os.path.exists(fpath):
        continue
    
    with open(fpath) as f:
        content = f.read()
    
    img_path = vals["img"]
    alt_text = vals["alt"]
    lqip_val = lqip_cache.get(vals["lqip"], "")
    
    new_img_block = f"""image:
  path: {img_path}
  alt: {alt_text}
  lqip: {lqip_val}"""
    
    # Replace existing image block
    content = re.sub(
        r'^image:\n(?:  .*\n?)*',
        new_img_block + '\n',
        content,
        flags=re.MULTILINE
    )
    
    with open(fpath, 'w') as f:
        f.write(content)
    
    print(f"  OK: {fname}")

print("\nAll front matter updated with alt text!")
