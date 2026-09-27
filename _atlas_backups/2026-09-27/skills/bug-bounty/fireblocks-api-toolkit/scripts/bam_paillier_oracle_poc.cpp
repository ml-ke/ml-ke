/**
 * BAM Paillier Oracle PoC — Append to test/cosigner/bam_test.cpp
 *
 * Run: cd build && cmake .. && make cosigner_test && \
 *      ./test/cosigner/cosigner_test "bam_oracle_poc" -s
 *
 * Expected: 34 assertions pass, showing distinguishable oracle gates.
 */

TEST_CASE("bam_oracle_poc")
{
    SECTION("oracle: encrypted_partial_sig probes produce distinguishable errors")
    {
        TestSetup testSetup;
        uuid_t uid;
        char keyid[UUID_STR_LEN] = {'\0'};
        char txid[UUID_STR_LEN] = {'\0'};
        const std::string setup_id(keyid);

        uuid_generate_random(uid);
        uuid_unparse(uid, keyid);
        uuid_generate_random(uid);
        uuid_unparse(uid, txid);

        elliptic_curve256_scalar_t hash;
        REQUIRE(RAND_bytes(hash, sizeof(hash)));

        elliptic_curve256_point_t X1, X2;
        bam_key_generation(setup_id, keyid, client_id, server_id,
            testSetup.server, testSetup.client, ECDSA_SECP256K1, X1, X2);

        fbc::signing_data data_to_sign = {{0},
            {{ fbc::byte_vector_t(&hash[0], &hash[sizeof(hash)]),
            { 44, 0, 0, 0, 0} }}};

        std::vector<fbc::bam_ecdsa_cosigner::server_signature_shared_data>
            server_shares;
        std::vector<fbc::bam_ecdsa_cosigner::client_partial_signature_data>
            partial_signatures;
        std::vector<fbc::recoverable_signature> full_signatures;
        cosigner_sign_algorithm signature_algorithm;

        REQUIRE_NOTHROW(testSetup.client.prepare_for_signature(keyid, txid, 0,
            server_id, client_id, data_to_sign, "", std::set<std::string>()));
        REQUIRE_NOTHROW(testSetup.server.generate_signature_share(keyid, txid, 0,
            server_id, client_id, ECDSA_SECP256K1, data_to_sign, "",
            std::set<std::string>(), server_shares));
        REQUIRE_NOTHROW(testSetup.client.compute_partial_signature(txid,
            server_shares, partial_signatures));
        REQUIRE(partial_signatures.size() == 1);

        size_t enc_size = partial_signatures[0].encrypted_partial_sig.size();
        size_t proof_size_actual = partial_signatures[0].sig_proof.size();

        std::cout << "\n  Reference: encrypted_sig=" << enc_size
                  << " bytes, proof=" << proof_size_actual << " bytes\n";

        auto probe = [&](const std::string& label,
                         fbc::byte_vector_t crafted_sig) -> std::string {
            auto sigs_copy = partial_signatures;
            sigs_copy[0].encrypted_partial_sig = std::move(crafted_sig);
            try {
                std::vector<fbc::recoverable_signature> sigs;
                cosigner_sign_algorithm algo;
                testSetup.server.verify_partial_signature_and_output_signature(
                    txid, client_id, sigs_copy, sigs, algo);
                return "PASS (decrypted + sig OK)";
            } catch (const fbc::cosigner_exception& e) {
                return std::string("ERR: ") + e.what();
            } catch (const std::exception& e) {
                return std::string("EXC: ") + e.what();
            }
        };

        // P1: All zeros — coprime check fails (gcd(0,n)=n)
        {
            auto result = probe("all zeros",
                fbc::byte_vector_t(enc_size, 0x00));
            std::cout << "  [P1] encrypted_sig = 0x00...00 -> "
                      << result << "\n";
            REQUIRE(result != "PASS (decrypted + sig OK)");
        }

        // P2: Value 1
        {
            fbc::byte_vector_t ct(enc_size, 0x00);
            ct.back() = 0x01;
            auto result = probe("trivial=1", ct);
            std::cout << "  [P2] encrypted_sig = 0x00...01 -> "
                      << result << "\n";
            REQUIRE(result != "PASS (decrypted + sig OK)");
        }

        // P3: Value 2
        {
            fbc::byte_vector_t ct(enc_size, 0x00);
            ct.back() = 0x02;
            auto result = probe("trivial=2", ct);
            std::cout << "  [P3] encrypted_sig = 0x00...02 -> "
                      << result << "\n";
            REQUIRE(result != "PASS (decrypted + sig OK)");
        }
    }

    SECTION("oracle: proof vs ciphertext corruption produce different errors")
    {
        TestSetup testSetup;
        uuid_t uid;
        char keyid[UUID_STR_LEN] = {'\0'};
        const std::string setup_id(keyid);
        uuid_generate_random(uid);
        uuid_unparse(uid, keyid);

        elliptic_curve256_scalar_t hash, hash2;
        REQUIRE(RAND_bytes(hash, sizeof(hash)));
        REQUIRE(RAND_bytes(hash2, sizeof(hash2)));

        elliptic_curve256_point_t X1, X2;
        bam_key_generation(setup_id, keyid, client_id, server_id,
            testSetup.server, testSetup.client, ECDSA_SECP256K1, X1, X2);

        // Probe A: valid ciphertext + corrupted proof (Gate 2 fail)
        {
            char txid[UUID_STR_LEN];
            uuid_generate_random(uid);
            uuid_unparse(uid, txid);

            fbc::signing_data data = {{0},
                {{ fbc::byte_vector_t(&hash[0], &hash[sizeof(hash)]),
                { 44, 0, 0, 0, 0} }}};
            std::vector<fbc::bam_ecdsa_cosigner::server_signature_shared_data>
                server_shares;
            std::vector<fbc::bam_ecdsa_cosigner::client_partial_signature_data>
                partial_sigs;

            REQUIRE_NOTHROW(testSetup.client.prepare_for_signature(keyid, txid,
                0, server_id, client_id, data, "", std::set<std::string>()));
            REQUIRE_NOTHROW(testSetup.server.generate_signature_share(keyid,
                txid, 0, server_id, client_id, ECDSA_SECP256K1, data, "",
                std::set<std::string>(), server_shares));
            REQUIRE_NOTHROW(testSetup.client.compute_partial_signature(txid,
                server_shares, partial_sigs));
            REQUIRE(partial_sigs.size() == 1);

            // Corrupt proof[0] — triggers Gate 2 (deserialize)
            if (!partial_sigs[0].sig_proof.empty())
                partial_sigs[0].sig_proof[0] ^= 0xFF;

            try {
                std::vector<fbc::recoverable_signature> sigs;
                cosigner_sign_algorithm algo;
                testSetup.server.verify_partial_signature_and_output_signature(
                    txid, client_id, partial_sigs, sigs, algo);
                FAIL("Corrupted proof should have been rejected");
            } catch (const fbc::cosigner_exception& e) {
                std::cout << "  [A] valid CT + corrupt proof -> ERR: "
                          << e.what() << "\n";
            }
        }

        // Probe B: corrupted ciphertext + valid proof (Gate 3/4 fail)
        {
            char txid_ref[UUID_STR_LEN];
            uuid_generate_random(uid);
            uuid_unparse(uid, txid_ref);

            fbc::signing_data ref_data = {{0},
                {{ fbc::byte_vector_t(&hash2[0], &hash2[sizeof(hash2)]),
                { 44, 0, 0, 0, 0} }}};
            std::vector<fbc::bam_ecdsa_cosigner::server_signature_shared_data>
                ref_shares;
            std::vector<fbc::bam_ecdsa_cosigner::client_partial_signature_data>
                ref_sigs;
            REQUIRE_NOTHROW(testSetup.client.prepare_for_signature(keyid,
                txid_ref, 0, server_id, client_id, ref_data, "",
                std::set<std::string>()));
            REQUIRE_NOTHROW(testSetup.server.generate_signature_share(keyid,
                txid_ref, 0, server_id, client_id, ECDSA_SECP256K1, ref_data,
                "", std::set<std::string>(), ref_shares));
            REQUIRE_NOTHROW(testSetup.client.compute_partial_signature(txid_ref,
                ref_shares, ref_sigs));
            REQUIRE(ref_sigs.size() == 1);

            char txid2[UUID_STR_LEN];
            uuid_generate_random(uid);
            uuid_unparse(uid, txid2);

            fbc::signing_data data2 = {{0},
                {{ fbc::byte_vector_t(&hash[0], &hash[sizeof(hash)]),
                { 44, 0, 0, 0, 0} }}};
            std::vector<fbc::bam_ecdsa_cosigner::server_signature_shared_data>
                server_shares2;
            std::vector<fbc::bam_ecdsa_cosigner::client_partial_signature_data>
                partial_sigs2;

            REQUIRE_NOTHROW(testSetup.client.prepare_for_signature(keyid,
                txid2, 0, server_id, client_id, data2, "",
                std::set<std::string>()));
            REQUIRE_NOTHROW(testSetup.server.generate_signature_share(keyid,
                txid2, 0, server_id, client_id, ECDSA_SECP256K1, data2, "",
                std::set<std::string>(), server_shares2));
            REQUIRE_NOTHROW(testSetup.client.compute_partial_signature(txid2,
                server_shares2, partial_sigs2));
            REQUIRE(partial_sigs2.size() == 1);

            // Flip one byte in the ciphertext, keep proof intact
            if (!partial_sigs2[0].encrypted_partial_sig.empty()) {
                partial_sigs2[0].encrypted_partial_sig[
                    ref_sigs[0].encrypted_partial_sig.size() / 2] ^= 0x01;
            }

            try {
                std::vector<fbc::recoverable_signature> sigs;
                cosigner_sign_algorithm algo;
                testSetup.server.verify_partial_signature_and_output_signature(
                    txid2, client_id, partial_sigs2, sigs, algo);
            } catch (const fbc::cosigner_exception& e) {
                std::cout << "  [B] corrupt CT + valid proof -> ERR: "
                          << e.what() << "\n";
            }
        }
    }
}
