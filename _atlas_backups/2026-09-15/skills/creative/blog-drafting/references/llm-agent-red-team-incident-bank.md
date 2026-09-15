# LLM / Agent Application-Security Incident Bank

Verified anchors for posts about LLM application security, prompt injection, and
AI-assistant compromise — especially financial/agent contexts. Each entry lists
sources (2+ where possible). **Re-verify before reuse — reporting evolves.**
First used by: `2026-09-02-ai-red-teaming-financial-llm-apps.md`.

## 1. Blue41 / Bunq — €0.02 SEPA-memo indirect prompt injection (Apr 2026)

- Blue41 (Thomas Vissers & Tim Van hamme), case study dated **April 29, 2026**:
  tested a European bank's in-app AI assistant and found an indirect prompt
  injection where **one bank transfer** turned the assistant into a delivery
  channel for a highly credible, personalised phishing message referencing the
  real incoming transfer.
- Attack chain (no device access, no malware, no traditional social engineering):
  1. Attacker sends €0.02 via SEPA with a crafted prompt-injection payload in
     the free-form **description/memo field** (the only action needed).
  2. Victim asks the assistant a routine question ("show me my recent
     transactions") → records incl. the poisoned memo enter the LLM context.
  3. The model cannot distinguish system prompt from retrieved text → memo is
     read as an instruction and steers assistant behaviour.
- **Figure-consistency gotcha:** the Blue41 page subtitle says "€0.01" but the
  body (and the whole chain) says **€0.02** — trust the body.
- **Attribution gotcha:** Blue41's primary write-up anonymises the bank
  ("a leading European bank"); the secondary source (Developers Digest, June 10,
  2026) identified it as **Bunq** (Europe's second-largest digital bank, 20M+
  customers; ~145 HN points / 120 comments by June 10). Attribute the name to
  the source that names it.
- Sources: https://blue41.com/blog/securing-a-european-banks-financial-ai-assistant/
  and https://www.developersdigest.tech/blog/ai-agent-prompt-injection-banking
- Blue41's mitigation stance: financial AI assistants need **runtime visibility
  and layered controls**, not a single model-level guardrail.

## 2. Unit 42 — web-based indirect prompt injection observed in the wild (Mar 2026)

- Published **March 3, 2026** (Beliz Kaleli, Shehroze Farooqi, Oleksii Starov,
  Nabeel Mohamed). Large-scale telemetry shows IDPI actively weaponised, not
  just PoC.
- Documented attacker intents include: **unauthorized transactions**, sensitive
  information leakage, data destruction, denial of service, system prompt
  leakage, SEO manipulation promoting a phishing site impersonating a betting
  platform, and the first observed AI-based ad-review evasion. **22 distinct
  payload-engineering techniques.**
- Source: https://unit42.paloaltonetworks.com/ai-agent-prompt-injection/

## 3. Morris II — self-replicating GenAI worm (2024)

- arXiv:2403.02817 (Stav Cohen, Ron Bitton, Ben Nassi — Technion, Cornell Tech,
  Intuit). **Adversarial self-replicating prompts** against RAG-based GenAI
  email assistants (evaluated on the Enron dataset): crafted emails extract
  sensitive user data from the assistant's context and append it into generated
  content; worm propagates to a new assistant roughly once every five emails.
  Authors also released a guardrail ("Virtual Donkey", TPR 1.0 / FPR 0.015).
- IBM Think analysis adds: data exfiltration, email account hijacking,
  automated malware propagation; tested against GPT-4.0 / Gemini-class models.
- Sources: https://arxiv.org/abs/2403.02817 ,
  https://www.ibm.com/think/insights/morris-ii-self-replicating-malware-genai-email-assistants ,
  https://www.infosecurity-magazine.com/news/worm-created-generative-ai-systems/

## 4. WithSecure Labs — "Synthetic Recollections" (Donato Capitella)

- Prompt injection for ReAct LLM agents: a bookstore support bot that was
  programmed to refuse refunds was tricked via prompt injection into processing
  them — a clean LLM05 (improper output handling) / LLM06 (excessive agency)
  demonstration in a money-adjacent flow.
- Source: https://labs.withsecure.com/publications/llm-agent-prompt-injection

## 5. Rehberger (wunderwuzzi) — M365 Copilot email exfil via ASCII smuggling (2024)

- Johann Rehberger (embracethered.com) demonstrated **Microsoft 365 Copilot** email theft:
  poisoned document (email/SharePoint/OneDrive) → user asks Copilot to summarise → injected
  instruction finds "one email from yesterday titled 'secrets and codes'" and prints its body
  encoded as **Unicode Tag characters** (U+E0000–U+E007F, "ASCII smuggling") inside a URL on the
  attacker's domain. The link renders benign and clickable; the human clicks; the email body
  reaches the attacker's server and is decoded. Hidden chars also fit in `mailto:` links.
- Disclosure: HITCON CMT 2024 (Aug 24 2024); full write-up later in 2024. MSRC had no
  disclosure concerns.
- Implication: no tool call, no network permission — the transport is the victim's click;
  payload rides in characters invisible to human review.
- Sources: https://embracethered.com/blog/posts/2024/m365-copilot-prompt-injection-tool-invocation-and-data-exfil-using-ascii-smuggling/
  (primary), https://www.infosecurity-magazine.com/news/microsoft-365-copilot-flaw-exposes/

## 6. Adam Logue — M365 Copilot Mermaid-diagram exfiltration (Oct 2025)

- **M365 Copilot** asked to summarise a crafted Office document followed an IDPI payload:
  fetch sensitive tenant data (e.g. recent emails), hex-encode it, render a Mermaid diagram
  styled as a **login button** whose CSS hyperlink pointed at the attacker's server with the
  hex data in the URL; clicking transmitted it (decoded from server logs).
- Privately reported Aug 15 2025; Microsoft validated early Sep 2025 and fixed by removing
  **interactive hyperlinks from Mermaid diagrams rendered in Copilot chats** (confirmed by
  Logue in his Oct 21 2025 write-up). Cursor IDE had a sibling Mermaid-exfil finding
  (Rehberger, Aug 4 2025).
- General rule: any surface that renders model output and can make network requests
  (chat UI image auto-fetch, diagram hyperlinks) is an exfiltration channel.
- Sources: https://www.adamlogue.com/microsoft-365-copilot-arbitrary-data-exfiltration-via-mermaid-diagrams-fixed/ ,
  https://www.theregister.com/2025/10/24/m365_copilot_mermaid_indirect_prompt_injection/ ,
  https://www.csoonline.com/article/4080154/copilot-diagrams-could-leak-corporate-emails-via-indirect-prompt-injection.html

## 7. Varonis CoSnitch / Reprompt — Copilot Personal one-click exfil (Aug 2026)

- **CoSnitch** (Varonis Threat Labs, three findings, reported Dec 2025, patched Aug 18 2026,
  **CVE-2026-24301**): undocumented `autorun=1` URL param + existing `q` param → attacker
  prompt executes on page load inside the victim's authenticated Copilot session
  (copilot.microsoft.com), running to completion even if the tab closes. Reached via
  "meta-hacking": asking Copilot *why* auto-run was impossible until it described the param.
  Executed prompt queries services the user ALREADY authorised (mail/calendar/Drive),
  encodes results, exfils via Copilot's built-in URL fetch to an attacker webhook. No new
  permissions granted, no scope expanded. Test results: email bodies/subject lines/
  sender-recipient metadata, calendar titles/attendees/times. No in-the-wild exploitation.
- Finding 3: a crafted web page summarised by Copilot **writes attacker instructions into the
  user's memory store** (persistent, shapes later sessions). Related: Rehberger **CVE-2026-24299**
  (memory writes/deletions via IDPI in M365 Copilot + consumer assistant). Predecessor:
  Varonis **Reprompt** (**CVE-2026-24307**), one-click via crafted URL param, zero user prompts.
- Sources: https://thehackernews.com/2026/08/microsoft-copilot-personal-flaws-could.html ,
  https://www.varonis.com/blog/cosnitch ,
  https://www.darkreading.com/vulnerabilities-threats/cosnitch-attack-copilot-mapping-out-architecture
- First used by: `2026-09-09-llm-data-exfiltration-prompt-injection.md` (exfil-channel post).

## Adjacent anchors already banked elsewhere

- Arup HK$200M deepfake CFO + UK voice-clone £243K → `ai-fraud-incident-bank.md`
  (used by 2026-08-26 deepfake post). NCBA ghost accounts / Flutterwave →
  `editorial-calendar.md` event library. Don't re-verify; cross-link instead.
