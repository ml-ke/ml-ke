# OWASP LLM Top 10 2025 — Reference Summary

## The Ranking

| Rank | Vulnerability | Change from 2023 |
|------|--------------|-------------------|
| LLM01 | Prompt Injection | ↔ (steady #1) |
| LLM02 | Sensitive Information Disclosure | ↑ from #6 |
| LLM03 | Supply Chain Vulnerabilities | ↑ from #5 |
| LLM04 | Data and Model Poisoning | renamed |
| LLM05 | Improper Output Handling | ↓ from #2 |
| LLM06 | Excessive Agency | new |
| LLM07 | Insecure Plugin/Tool Design | new |
| LLM08 | Excessive Dependencies on AI-generated Code | new |
| LLM09 | Model Denial of Service | ↔ |
| LLM10 | Model Theft | new |

## Key Real-World Incidents (2024-2026)

### Prompt Injection (LLM01)
- **Slack AI (Aug 2024)** — PromptArmor: attacker posts injection payload in public channel, Slack AI exfiltrates private channel data via image URL. Mitigation: CSP. Root issue: instructions and data share the same channel.
- **CVE-2024-5184** — LLM email assistant: injection in email subject line causes email forwarding, calendar modification, evidence deletion.
- **GitHub Agent Cross-Repo Theft (2025)** — malicious Issue body injects coding agent, exfiltrates private repo source code.

### Jailbreaks (related to LLM01)
- **DAN (Do Anything Now)** — 14+ variants, roleplay-based bypass. Era: 2022-2023.
- **Parseltongue** — exploits BPE tokenizer quirks to create adversarial tokens.
- **GODMODE** — developer-mode emulation, multi-step prompt.
- **ULTRAPLINIAN** — chain-of-thought jailbreak.
- **WormGPT / FraudGPT** — dark web LLMs fine-tuned without safety training.

### Data Poisoning (LLM04)
- **torchtriton (CVE-2022-45907)** — dependency confusion on PyPI, stole SSH keys and cloud credentials.
- **Hugging Face Pickle RCE** — default pickle format in PyTorch allows arbitrary code execution during model loading. Mitigation: Safetensors (~60% adoption as of 2026).
- **2025 Chinese LLM Backdoors** — trigger-phrase-activated behavior changes in model weights, invisible to standard scanning.

### Excessive Agency (LLM06)
- **Auto-GPT buying domains** (2023) — agent spent real money without confirmation.
- **LangChain SSRF (CVE-2023-36085)** — agent exposed internal services.
- **AI agents deleting databases** — multiple unreported cases.

### Supply Chain (LLM03)
- **PyTorch dependency confusion** — package squatting on PyPI.
- **RAG poisoning via Wikipedia** — subtle edits to public knowledge bases inject misinformation into RAG pipelines.
- **Model provenance** — 40% of Hugging Face models still use pickle format.

## Defense Patterns

| Vulnerability | Key Mitigations |
|---------------|-----------------|
| Prompt Injection | Prompt isolation, input sanitization, output validation, least privilege |
| Jailbreaks | Perplexity detection, input normalization, adversarial training, safety classifiers |
| Data Poisoning | Safe serialization (Safetensors), weight scanning, provenance verification |
| Excessive Agency | Human-in-the-loop, rate limits, scoped tool access, audit logging |
| Supply Chain | Hash verification, dependency pinning, sandboxed model loading |

## Sources
- OWASP Top 10 for LLMs 2025: https://owasp.org/www-project-top-10-for-large-language-model-applications/
- OWASP Gen AI Incident Round-up Q2 2025
- PromptArmor Research: https://promptarmor.com/resources/
- Wald AI: Gen AI Security Breaches Timeline 2023-2025
