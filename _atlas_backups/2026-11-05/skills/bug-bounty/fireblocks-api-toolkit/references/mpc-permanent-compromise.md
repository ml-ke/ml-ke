# Permanent Compromise: Version Downgrade + Missing Aux Key Rotation

## The Two-Failure Architecture

### Failure 1: Version Check Is One-Directional
- `cmp_setup_service.cpp:145-150`: `if (version > MPC_PROTOCOL_VERSION) throw;` — only blocks UPGRADE
- `cmp_ecdsa_online_signing_service.cpp:145-149`: same pattern
- `MPC_MIN_SUPPORTED_PROTOCOL_VERSION = 2` (`mpc_globals.h:12`) — dead code, zero grep hits in src/
- Effect: version 10 accepted silently, all CMP ops at v<11

### Failure 2: Refresh Doesn't Rotate Auxiliary Keys
- `cmp_offline_refresh_service.cpp:152-164`: ONLY updates ECDSA key share, no Paillier/RP regeneration
- `cmp_offline_refresh_service.cpp:223-224`: `load_auxiliary_keys(key_id, aux)` — same keys as setup
- No BAM refresh code exists anywhere in src/
- Effect: Paillier/RP keys are generated once, fixed forever

### Combined: Rotation-Surviving Permanent Compromise
- ECDSA key rotation ≠ Paillier key rotation
- Version downgrade persists in metadata after refresh
- Attacker who recovers λ has permanent decryption capability
- Only fix: generate entirely new wallet

## PoC Verification
- `attacker_persistent_compromise` test case (11/11 assertions)
- Assets 1-3: compiled assertion (version downgrade, encoding gap, FS seed)
- Assets 4-5: source code confirmed (no Paillier rotation, no BAM refresh)
- Asset 6: reasoned conclusion

## VRT
Cryptographic Weakness → Insecure Implementation → Missing Cryptographic Step (Varies baseline)

## References
- src/common/cosigner/cmp_offline_refresh_service.cpp:152-164
- src/common/cosigner/cmp_offline_refresh_service.cpp:223-224
- src/common/cosigner/cmp_setup_service.cpp:145-150
- grep -rn "bam.*refresh\|refresh.*bam" src/ → zero results
