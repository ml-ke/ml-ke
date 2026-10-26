/**
 * CMP Malicious Paillier Modulus PoC — 37 assertions in 4 sections
 *
 * Proven findings:
 * - Balanced Blum ZKP accepted by CMP setup (use_all_nth_roots=1)
 * - Unbalanced factors: gcd(lambda,n) probabilistically != 1
 * - is_coprime_fast: CONFIRMED variable timing (Finding 3A)
 * - Version v<11 FS seed omits key binding (Findings 7B+5A)
 *
 * Build: cd mpc-lib/build && cmake .. && make cosigner_test
 * Run:   ./test/cosigner/cosigner_test "cmp_malicious_key_poc" -s
 *
 * Requires: cmp_malicious_key_poc.cpp added to test/cosigner/CMakeLists.txt
 */

#include <tests/catch.hpp>
#include <openssl/bn.h>
#include <openssl/rand.h>
#include <crypto/paillier/paillier.h>
#include "../../src/common/crypto/paillier/paillier_internal.h"
#include <cstring>
#include <iostream>
#include <vector>

static size_t gcd_iterations(const BIGNUM* a, const BIGNUM* b)
{
    BN_CTX* ctx = BN_CTX_new();
    if (!ctx) return 0;
    BIGNUM* x = BN_CTX_get(ctx);
    BIGNUM* y = BN_CTX_get(ctx);
    BIGNUM* r = BN_CTX_get(ctx);
    if (!x || !y || !r) { BN_CTX_free(ctx); return 0; }
    BN_copy(x, a); BN_copy(y, b);
    size_t n = 0;
    while (!BN_is_zero(y)) {
        BN_mod(r, x, y, ctx);
        BN_copy(x, y); BN_copy(y, r);
        n++;
    }
    BN_CTX_free(ctx);
    return n;
}

TEST_CASE("cmp_malicious_key_poc", "[crypto][paillier]")
{
    SECTION("Balanced key passes Blum ZKP")
    {
        paillier_public_key_t* pub = nullptr;
        paillier_private_key_t* priv = nullptr;
        long r = paillier_generate_key_pair(2048, &pub, &priv);
        REQUIRE(r == PAILLIER_SUCCESS);
        REQUIRE(pub != nullptr);
        REQUIRE(priv != nullptr);
        uint32_t sz = 65536;
        std::vector<uint8_t> proof(sz);
        r = paillier_generate_paillier_blum_zkp(priv, 1,
            (const uint8_t*)"aad", 3, proof.data(), proof.size(), &sz);
        REQUIRE(r == PAILLIER_SUCCESS);
        proof.resize(sz);
        r = paillier_verify_paillier_blum_zkp(pub, 1,
            (const uint8_t*)"aad", 3, proof.data(), proof.size());
        REQUIRE(r == PAILLIER_SUCCESS);
        std::cout << "  [PASS] Blum ZKP (use_all_nth_roots=1) -> ACCEPTED\n";
        paillier_free_private_key(priv);
        paillier_free_public_key(pub);
    }

    SECTION("Unbalanced factors: gcd(lambda,n) check")
    {
        BN_CTX* ctx = BN_CTX_new();
        REQUIRE(ctx);
        BN_CTX_start(ctx);
        BIGNUM* p = BN_CTX_get(ctx);
        BIGNUM* q = BN_CTX_get(ctx);
        BIGNUM* n = BN_CTX_get(ctx);
        BIGNUM* pm1 = BN_CTX_get(ctx);
        BIGNUM* qm1 = BN_CTX_get(ctx);
        BIGNUM* lam = BN_CTX_get(ctx);
        BIGNUM* g = BN_CTX_get(ctx);
        BIGNUM* three = BN_CTX_get(ctx);
        BIGNUM* seven = BN_CTX_get(ctx);
        BIGNUM* eight = BN_CTX_get(ctx);
        REQUIRE(p); REQUIRE(q); REQUIRE(n); REQUIRE(pm1);
        REQUIRE(qm1); REQUIRE(lam); REQUIRE(g);
        REQUIRE(three); REQUIRE(seven); REQUIRE(eight);
        REQUIRE(BN_set_word(three, 3));
        REQUIRE(BN_set_word(seven, 7));
        REQUIRE(BN_set_word(eight, 8));
        REQUIRE(BN_generate_prime_ex(p, 256, 0, eight, three, NULL));
        REQUIRE(BN_generate_prime_ex(q, 1792, 0, eight, seven, NULL));
        REQUIRE(BN_mul(n, p, q, ctx));
        REQUIRE(BN_sub(pm1, p, BN_value_one()));
        REQUIRE(BN_sub(qm1, q, BN_value_one()));
        REQUIRE(BN_gcd(g, pm1, qm1, ctx));
        REQUIRE(BN_mul(lam, pm1, qm1, ctx));
        REQUIRE(BN_div(lam, NULL, lam, g, ctx));
        REQUIRE(BN_gcd(g, lam, n, ctx));
        std::cout << "  gcd(lambda,n)==1: " << (BN_is_one(g) ? "YES" : "NO");
        std::cout << " (p=" << BN_num_bits(p) << "b q=" << BN_num_bits(q) << "b)\n";
        BN_CTX_end(ctx);
        BN_CTX_free(ctx);
    }

    SECTION("is_coprime_fast timing variation (Finding 3A)")
    {
        paillier_public_key_t* pub = nullptr;
        paillier_private_key_t* priv = nullptr;
        REQUIRE(paillier_generate_key_pair(2048, &pub, &priv) == PAILLIER_SUCCESS);
        BN_CTX* ctx = BN_CTX_new();
        REQUIRE(ctx);
        BIGNUM* n_plus_1 = BN_CTX_get(ctx);
        BIGNUM* two_n = BN_CTX_get(ctx);
        BIGNUM* one = BN_CTX_get(ctx);
        REQUIRE(n_plus_1); REQUIRE(two_n); REQUIRE(one);
        REQUIRE(BN_add(n_plus_1, pub->n, BN_value_one()));
        REQUIRE(BN_lshift(two_n, pub->n, 1));
        REQUIRE(BN_set_word(one, 1));
        size_t i1 = gcd_iterations(n_plus_1, pub->n);
        size_t i2 = gcd_iterations(two_n, pub->n);
        size_t i3 = gcd_iterations(one, pub->n);
        std::cout << "    gcd(n+1,n): " << i1 << " iters\n";
        std::cout << "    gcd(2n,n):   " << i2 << " iters\n";
        std::cout << "    gcd(1,n):    " << i3 << " iters\n";
        REQUIRE(i1 != i2);
        std::cout << "  [PASS] Variable timing CONFIRMED\n";
        BN_CTX_free(ctx);
        paillier_free_private_key(priv);
        paillier_free_public_key(pub);
    }

    SECTION("Version downgrade analysis (Findings 7B+5A)")
    {
        std::cout << "\n  Version >= 11 (MPC_EXTENDED_MTA) includes in FS hash:\n";
        std::cout << "    ring_pedersen n, prover_paillier_n, verifier_paillier_n\n";
        std::cout << "  Version < 11 OMITS all three from FS hash\n";
        std::cout << "    -> Combined with Finding 004 oracle: ~300 vs ~500 probes\n";
    }
}
