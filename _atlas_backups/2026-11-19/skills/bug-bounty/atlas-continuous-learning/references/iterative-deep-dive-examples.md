# Iterative Deep-Dive — Worked Examples

Detail moved out of SKILL.md for progressive disclosure (loaded on demand).

**Real example — Nutaku to AdultForce pivots (Jun 2026, 7 iterations):**\n1. Probed Nutaku gateway creds → gateway down\n2. Probed Nutaku _xd API → login protected, user endpoints 404\n3. Probed Nutaku OSAPI → root responds but REST paths 404\n4. Probed Nutaku signup, members pages → CSRF/recaptcha locked\n5. Searched Nutaku JS files → found AdultForce URL embedded\n6. Probed AdultForce → found /api/config publicly accessible (operational messages)\n7. Probed /api/site → 155 brand sites with ProBiller billing IDs, S3 paths, GA accounts discovered (real finding)


**Real example — Visma → Auth0 → Torfs → Nexuzhealth (Jun 2026):**
1. Visma: Probed AI Assistant, OIDC, MCP server — blocked by credential requirements
2. Pivoted to Torfs (SFCC): Mapped full OCAPI/SLAS/SCAPI surface, registered account, extracted JWT with 26 scopes
3. Pivoted to Nexuzhealth: Mapped LiquidFiles, SimpleSAMLphp IdP, Jira dashboards, XERO PACS viewer
4. Each pivot was informed by the prior target's auth pattern (SAML/OIDC similarities)
5. Key learning: Multi-target parallel recon reveals patterns that single-target focus misses


**Real example — Visma documentation search (Jun 2026):**
1. Probed VismaOnline stage → login gated, no self-registration on identity server
2. Probed testing.maventa.com → registration blocked by valid org number requirement
3. Searched official Visma bug bounty docs → found getting-started PDFs on Azure blob storage (`vismabugbountyprod.z16.web.core.windows.net`)
4. PDFs revealed: student signup URL `https://admin.stage.vismaonline.com/Customer/StudentSignup.aspx?uilang=en` with training code `g004t`
5. PDFs also revealed: Developer Portal test app naming conventions, OAuth2 client_credentials test APIs, and scope documentation


**Where to find official program docs (when registration is blocked):**
- Search `site:<program-domain> getting started` or `site:<program-domain> test account`
- Search `<program-name> bug bounty getting started pdf` via web search
- Check the program's GitHub repos for README/SECURITY.md files with test instructions
- Check Azure blob storage: often `<company>bugbountyprod.z16.web.core.windows.net/*.pdf`
- Check program support knowledge bases (e.g. `support.maventa.fi`, `community.visma.com`)
- Check the program's developer portal for sandbox/test environment documentation
- Search for the program name on FireBounty, which sometimes links to test credentials docs

