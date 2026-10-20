# Fact-Checking Workflow for Bug Reports

## Why This Matters

Bugcrowd triage will verify every claim against source code. A single factual error (e.g., "constant X means Y" when X is never checked) can get the entire report rejected or cause the researcher to lose credibility.

## Pre-Submission Checklist

### 1. Trace Every Source Code Reference

For each `source:path/to/file:LINE` in your report:

```bash
# Verify the file exists at that path
ls ~/Dev/mpc-lib/path/to/file

# Read the actual line
sed -n 'LINEp' ~/Dev/mpc-lib/path/to/file

# Check surrounding context
sed -n 'LINE-5,LINE+5p' ~/Dev/mpc-lib/path/to/file
```

### 2. Verify Constants Are Actually Used

A named constant in a header is NOT evidence of an implemented feature:

```bash
# Check if the constant is ever referenced in source
grep -rn "CONSTANT_NAME" ~/Dev/mpc-lib/src/

# If it returns zero results, the constant is DEAD CODE
# Do NOT claim its intended behavior exists
```

**Real example:** `MPC_DONT_ENCRYPT_MTA_RESPONSE_PROTOCOL_VERSION = 3` is defined in `mpc_globals.h` but NEVER checked in any `.c` or `.cpp` file. Claiming "version 2 doesn't encrypt MTA responses" based on this constant name alone is WRONG.

### 3. Trace the Actual Execution Path

Don't assume what a constant name means. Trace how it's actually used:

```bash
# Find all references to the constant
grep -rn "CONSTANT_NAME" ~/Dev/mpc-lib/

# For version checks, verify what happens at each version
# Look for "if (version >= X)" patterns
grep -rn "version >=" ~/Dev/mpc-lib/src/ | grep -v test
```

### 4. Verify Different Primitives Before Comparing Key Sizes

Before claiming "protocol A uses X-bit keys but B uses Y-bit":

```bash
# Check which FUNCTION generates the keys
grep -B5 "generate_key" path/to/file

# Is it `paillier_generate_key_pair` or `paillier_commitment_generate_private_key`?
# These are DIFFERENT cryptosystems with different key size requirements.
```

### 5. Assess Severity Honestly

Start from the LOWEST likely severity and build up:

| Observation | Starting Severity | Why |
|------------|-------------------|-----|
| Design/config choice (e.g., default protocol) | P5 | Operational decision, not a bug |
| Theoretical parameter bound (e.g., 'knife-edge') | P5 | Needs demonstrated attack |
| Weak ZKP parameter (e.g., simpler Fiat-Shamir seed) | P4 | Depends on attack context |
| Demonstrated oracle (e.g., distinguishable errors) | P3-P2 | Depends on exploitability |
| Full key recovery chain with PoC | P1-P2 | Requires <1000 aborts for P1 |

### 6. "As an Attacker I Would..." — Does This Actually Work?

Run through the attack mentally:

- **Step 1**: Is there a viable entry point? (network access? malicious cosigner role?)
- **Step 2**: Can I actually observe the claimed oracle behavior?
- **Step 3**: Does the math work? (e.g., can ~300 queries actually recover a Paillier key?)
- **Step 4**: What blocks me? (nonce tracking? rate limiting? TLS?)
- **Step 5**: Is the impact real? (key recovery = yes, info disclosure = maybe)

If any step has a gap, note it. Don't claim a chain that doesn't close.

## Red Flags That Need Re-Verification

- A `constexpr` or `#define` that sounds like a security feature → check if it's used
- Two numbers compared across different files → check if they're the same primitive
- "This was blogged about" → verify the blog actually says what you claim
- Version-based behavior → trace the actual `if (version >= X)` branches
- Any claim beginning with "this means" followed by speculation → find the code that proves it

## References

- Source tree: `~/Dev/mpc-lib/`
- Build: `cd ~/Dev/mpc-lib/build && cmake .. -DCMAKE_BUILD_TYPE=Debug && make -j$(nproc)`
- Test runner: `./test/cosigner/cosigner_test "test_name"`
- Full text search: `grep -rn "pattern" ~/Dev/mpc-lib/src/`
