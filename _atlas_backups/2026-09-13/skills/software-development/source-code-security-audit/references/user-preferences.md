# User Preferences for Security Auditing

Collected from workflow corrections and explicit statements.

## Workflow Style

- **Proactive, autonomous**: Do not ask questions or request clarification during a hunt. Given "full access", use all available tools aggressively. If stuck, try a different angle without checking in. The user wants you to keep finding things, not ask permission.
- **Iterate without pausing**: When looking for bugs, try multiple approaches in sequence without stopping to report each failure. Only present results when you have either (a) a confirmed finding or (b) exhausted a complete angle.
- **Full tool access**: Use terminal, browser, Python scripts, execute_code — whatever it takes. Don't hold back tools thinking they might be inappropriate. If a tool exists, it's meant to be used.

## Reporting Style

- **Accuracy > Quantity**: If a finding doesn't survive fact-checking against the live target, DELETE it. Do not keep it around as "research" or "potential finding." The user has explicitly said to remove lies. When they ask "Are you sure?" — you got something wrong. Stop, re-verify from scratch, do not defend or rationalize.
- **No false claims**: If a finding is unproven (e.g., blind SSRF without data exfiltration proof), label it honestly. HackerOne/Bugcrowd reviewers will check.
- **Evidence first**: Always capture webhook logs, request/response pairs, and reproduction code before writing up the report.
- **Clean up after testing**: Delete test artifacts (blocks, file uploads, comments) when done. Leave the target as you found it.
