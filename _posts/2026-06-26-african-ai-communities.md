---
title: "African AI Communities: Masakhane, Lacuna Fund, and Local Dataset Initiatives"
date: 2026-06-26 00:00:00 +0300
categories: [Machine Learning, AI in Africa]
tags: [africa, masakhane, lacuna-fund, african-datasets, community]
image:
  path: /assets/img/cover-african-ai-communities.webp
  alt: African AI meetup with diverse participants presenting research
---

## The Power of Community-Driven AI

If you read Western tech media, you might believe that AI progress comes from a handful of Silicon Valley labs with unlimited GPU clusters. The reality in Africa is completely different. Progress comes from WhatsApp groups where researchers share Colab credits, from weekend hackathons running on university power backups, and from communities that have organized around the simple truth: **African AI must be built by Africans**.

This post covers the key communities, initiatives, and funding sources that make up the African AI ecosystem — and how you can get involved.

## Masakhane: Decolonizing NLP, One Language at a Time

### Origins and Mission

Masakhane was founded in 2019 at the Deep Learning Indaba in Nairobi by a group of African NLP researchers frustrated by the lack of representation for African languages in NLP research. The name comes from the isiZulu and isiXhosa word *ukusakha* meaning "to build" with the reflexive prefix *-ma-*, giving "we build together" or "we build ourselves."

Their mission: **to build NLP research capacity for African languages, by Africans**.

### What They've Achieved

| Metric | As of 2025 |
|--------|------------|
| Members | 2,500+ across 50+ countries |
| Languages covered | 50+ African languages |
| Peer-reviewed publications | 80+ |
| Datasets released | 30+ (including NER, sentiment, translation) |
| Workshops organized | 20+ (collocated with major NLP conferences) |

### Key Research Contributions

1. **Masakhane Machine Translation Benchmark**: Parallel corpora and baselines for 15 African language pairs, using standardized evaluation protocols that account for dialect variation.

2. **AfriSenti**: A sentiment analysis dataset covering 14 African languages with 110,000+ annotations. Each annotation includes a confidence score from multiple native speakers — addressing the problem of inter-annotator agreement in low-resource settings.

3. **MasakhaNER 2.0**: Named entity recognition for 20 African languages. The initial dataset covered 10 languages; the community expanded it through a distributed annotation effort involving linguists and native speakers across the continent.

4. **MasakhaNEWS**: News topic classification dataset for 16 African languages, demonstrating that cross-lingual transfer within language families (Bantu, Cushitic, Mande) outperforms cross-lingual transfer from English.

### How They Work

Masakhane operates through a distributed, volunteer-driven model:

- **Paper Discussion Groups**: Weekly video calls focused on a specific paper, alternating between foundational NLP papers and African-specific research
- **Sprint Weeks**: Intensive 1-2 week periods where members form small teams to work on specific projects — dataset creation, baseline implementations, or paper writing
- **Mentorship Program**: Senior researchers (often African diaspora academics) mentor early-career members through the full research lifecycle: idea → experiment → paper submission
- **WhatsApp/Slack Community**: Real-time discussion, problem-solving, and resource sharing. The "GPU request" channel has members offering Colab credits or access to university clusters

### How to Contribute

- **Join the mailing list**: masakhane@googlegroups.com
- **Participate in sprints**: Follow @MasakhaneNLP on X (Twitter) for sprint announcements
- **Contribute data**: If you speak an African language, you can help annotate datasets — no ML experience required
- **Submit papers**: Masakhane-affiliated papers are welcome at venues like AfricaNLP (workshop at ICLR), EACL AfricaNLP, and the Deep Learning Indaba proceedings

## Lacuna Fund: Funding the Data We Need

### The Problem Lacuna Fund Addresses

Most labeled datasets are created by large tech companies for high-resource languages and wealthy markets. The result: excellent AI for English-speaking users in the Global North, and almost nothing for Swahili speakers in rural Tanzania or Wolof speakers in Senegal.

Lacuna Fund, launched in 2020, is a collaborative fund that provides grants to create, extend, and maintain labeled datasets for low-resource contexts — with a strong focus on Africa.

### What They Fund

| Grant Round | Focus Area | Examples |
|-------------|-----------|----------|
| Round 1 (2020) | Language datasets | 9 projects, including Hausa NER, Swahili speech, Amharic sentiment |
| Round 2 (2022) | Agriculture datasets | 11 projects, including cassava disease, crop yield, soil health |
| Round 3 (2023) | Health datasets | 8 projects, including maternal health NLP, disease diagnosis |
| Round 4 (2024) | Climate & environment | 6 projects, including weather prediction, deforestation detection |

### How Lacuna Fund Is Different from Traditional Grantmaking

1. **Community-led**: Grant applicants must demonstrate that the dataset will be created *by* people from the target context, not just *about* them. Co-creation is mandatory.

2. **Open by default**: All funded datasets are released under open licenses (CC-BY, CC-BY-SA, or equivalent).

3. **Sustainability built in**: Grants include funding for dataset maintenance, not just creation. A dataset that's created and then abandoned is worse than no dataset.

4. **Ethical guidelines**: Funded projects must follow the Lacuna Fund Ethical Guidelines, which include:
   - Informed consent from data subjects
   - Fair compensation for annotators
   - Data minimization (collect only what's needed)
   - Privacy protection by design

### Noteworthy Lacuna Fund Projects

- **Africa AI for Agriculture**: Satellite + ground-truth dataset for crop type mapping in Kenya and Tanzania (10,000+ geo-tagged field images)
- **Kinyarwanda and Kirundi Speech**: 2,000 hours of transcribed speech for two closely related Bantu languages
- **Ethiopian Crop Diseases**: 50,000+ labeled images of teff, coffee, and maize diseases

## Deep Learning Indaba

The Indaba (isiZulu for "gathering" or "conference") is the flagship event for African ML. Founded in 2017 by Shakir Mohamed, Ulrich Paquet, and others, it has grown from 200 attendees to 600+ in-person and 3,000+ virtual participants annually.

### What Makes the Indaba Special

- **Rotating location**: The conference moves to a different African country each year (Johannesburg, Nairobi, Accra, Tunis, Dakar, Kigali...). This deliberately spreads capacity-building across the continent.
- **IndabaX**: 30+ independently organized local events in countries that can't host the main conference. These are affordable (often free) and accessible to local students.
- **Practical focus**: The Indaba's main program includes hands-on tutorials on topics like "Building NLP systems for low-resource languages" and "Deploying models on mobile devices with TFLite."
- **Travel scholarships**: The Indaba covers travel, accommodation, and registration for 80%+ of attendees, funded by sponsors including Google DeepMind, Meta, and the Allen Institute for AI.

## Data Science Nigeria (DSN)

Founded by Dr. Olubayo Adekanmbi, DSN is the largest AI education and research community in West Africa.

### Impact Numbers

- **100,000+** individuals trained through bootcamps and online courses
- **50+** university partnerships across Nigeria
- **AI Hub** network: Physical AI labs at partner universities with GPU access
- **$1M+** in AI research grants distributed to Nigerian academics

DSN's hackathon program deserves special mention. Annual hackathons attract 3,000+ participants and have spawned real startups: Crop2X (AI for agriculture), MediPredict (healthcare analytics), and CashToken (fintech fraud detection).

## Other Communities Worth Knowing

### Zindi
Africa's data science competition platform. Unlike Kaggle, Zindi problems are specifically African: predicting malaria outbreak risk in Senegal, classifying informal settlements in satellite imagery of Nairobi, or forecasting mobile money transaction volumes in Tanzania. With 50,000+ users from 190+ countries (but heavily weighted toward Africa), Zindi has become the primary way African data scientists build portfolios and get hired. The platform also runs hiring challenges where companies recruit directly from competition leaders.

### AI Media Group (South Africa)
An Africa-wide professional network for AI practitioners with regional chapters in South Africa, Nigeria, Kenya, and Ghana. They run the AI Expo Africa conference and the "AI for Good in Africa" working groups.

### Qubit by Qubit (Quantum Africa)
While quantum ML is still early-stage, this community is building Africa's quantum ML capacity through online courses and research collaborations, recognizing that Africa can't afford to miss yet another technology wave.

### African Masters of Machine Learning (AMML)
A fellowship program and community that supports African ML practitioners through mentorship, skill-building, and networking opportunities.

## How to Build Your Own Local AI Community

Inspired to start something in your city or university? Here's advice from the Masakhane founders:

1. **Start small**: A WhatsApp group with 5 people is more valuable than a Slack channel with 200 lurkers.
2. **Create value immediately**: "Let's all read papers" doesn't work. Start with a specific project — "Let's build a Twi chatbot" — and let the learning follow.
3. **Meet where people are**: In Africa, that means WhatsApp, Telegram, and in-person meetups (where WhatsApp groups originated). Don't assume your community will adopt Discord.
4. **Deal with compute inequality**: Your group will have members with access to GPUs and members who can only run Colab. Design projects that accommodate both. GPU-rich members handle training; others handle data collection, annotation, and evaluation.
5. **Publish or perish lite**: Even a blog post or a short paper at an African workshop is valuable. Celebrate every output to maintain momentum.

## The Funding Landscape

If you need funding for an African AI project, here are the main sources:

| Source | Typical Grant Size | Best For |
|--------|-------------------|----------|
| Lacuna Fund | $50K-$150K | Dataset creation |
| IDRC / AI4D Africa | $20K-$100K | Research projects |
| Google Research Awards | $10K-$50K | Academic research |
| Mozilla Common Voice | $5K-$30K | Speech data collection |
| Deep Learning Indaba Grants | $1K-$10K | Small community projects |
| Local AI associations | $500-$5K | Meetups, hackathons |

## The Big Picture

The African AI community has achieved remarkable things with minimal resources. Masakhane has produced more peer-reviewed NLP research on African languages than all of the world's corporate AI labs combined. Lacuna Fund has jumpstarted a dataset creation ecosystem that the World Bank estimates will unlock $3B in AI value for low-resource contexts by 2030. The Indaba has trained a generation of African ML researchers who now work at DeepMind, Google Brain, Meta AI, and OpenAI.

But the work is far from over. Of the 2,000+ African languages, fewer than 50 have any digital NLP resources at all. GPU access remains a bottleneck. And too much African AI talent still leaves the continent because there aren't enough local research positions.

The communities described here are the scaffolding for a self-sustaining African AI ecosystem. They need more contributors, more funding, and more real-world deployments. The best time to join was 2019. The second best time is now.

*Next in the series: AI for agriculture — real ML applications in Kenyan farming, from crop disease detection to yield prediction.*
