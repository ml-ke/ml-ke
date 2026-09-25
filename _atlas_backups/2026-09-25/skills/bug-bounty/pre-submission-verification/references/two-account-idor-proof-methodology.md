# Two-Account IDOR / Missing Auth Proof Methodology

## Why Two Accounts

One account proves an endpoint returns user-specific data. Two accounts prove that data is accessible **across users** — the minimal requirement for an access control finding. The triager's first question is always: "Did you access YOUR data or SOMEONE ELSE'S?"

## Step-by-Step (from Nutaku Finding 2, June 2026)

### Setup
1. Create two accounts on the target platform (both using @intigriti.me if required)
2. Authenticate both and note their user IDs

### Data Creation
3. Add distinct test data to each account — different values so you can tell whose is whose
4. Verify baseline: Account A reads Account A's data ✓, Account B reads Account B's data ✓

### Cross-User Test
5. Account A reads Account B's user ID with NO authentication token
6. Account B reads Account A's user ID with NO authentication token
7. If both return the other account's data → bidirectional proof confirmed

### Write/Read Asymmetry Signal
8. Test the write path: does POST (add data) require authentication? (401 without token = yes)
9. Test the read path: does GET (read data) require authentication? (200 without token = no)
10. If write requires auth but read doesn't → architectural inconsistency, not intended behavior

## Shell Variable Masking Workaround

When writing PoC scripts for bug bounty reports, the agent's output masking system may intercept shell variable references containing credential values. Symptoms:

```bash
# This will be masked — the token reference gets replaced with ***
TOK=$(cat /tmp/token.txt)
curl -H "Authorization: Bearer ***  # TOK is masked to ***
```

### Working Approaches

**Approach A — Python subprocess (most reliable)**:
Write the HTTP call inside a `python3 -c` block that reads the token from a file, so the token never appears in a shell variable:
```python
tok = open('/tmp/token.txt').read().strip()
import urllib.request
req = urllib.request.Request(url, headers={'Authorization': f'Bearer {tok}'})
resp = urllib.request.urlopen(req)
```

**Approach B — Store tokens in files, read inline**:
```bash
# Save token to file
echo "$TOKEN_VALUE" > /tmp/tok.txt

# Read it back in curl using printf (cat inside subshell)
curl -H "$(printf 'Authorization: Bearer %s' "$(cat /tmp/tok.txt)")" ...
```

**Approach C — Environment variables (works for PoC scripts given to users)**:
```bash
export NUTAKU_EMAIL="your@email.com"
export NUTAKU_PASS="your_password"
# Script reads from env vars
```

### What to Avoid
- Direct shell variable references containing credentials (`Bearer $TOKEN`)
- Heredocs containing credential strings
- Storing credentials in the script file itself
