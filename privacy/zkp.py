import numpy as np
import hashlib
import secrets

# ==============================
# SCHNORR ZKP PARAMETERS
# ==============================

# Large prime (Mersenne prime)
P = 2**127 - 1
G = 2


# ==============================
# PROVER (BANK SIDE)
# ==============================
class ZKPProver:
    def __init__(self, gradient):
        # Convert gradient → secure hash → integer secret
        gradient_bytes = gradient.astype(np.float32).tobytes()
        hash_hex = hashlib.sha256(gradient_bytes).hexdigest()

        # Use FULL hash (not truncated)
        self.secret = int(hash_hex, 16) % P

    def commit(self):
        # Secure random nonce
        self.nonce = secrets.randbelow(P - 1)

        # Commitment: G^nonce mod P
        self.commitment = pow(G, self.nonce, P)
        return self.commitment

    def respond(self, challenge):
        # response = nonce + challenge * secret (mod P-1)
        return (self.nonce + challenge * self.secret) % (P - 1)

    def get_public_key(self):
        # Public key = G^secret mod P
        return pow(G, self.secret, P)


# ==============================
# VERIFIER (SERVER SIDE)
# ==============================
class ZKPVerifier:
    def generate_challenge(self):
        # Secure random challenge
        self.challenge = secrets.randbelow(P - 1)
        return self.challenge

    def verify(self, commitment, response, public_key):
        # Verify:
        # G^response == commitment * public_key^challenge (mod P)

        lhs = pow(G, response, P)
        rhs = (commitment * pow(public_key, self.challenge, P)) % P

        return lhs == rhs


# ==============================
# FULL ZKP FLOW (ONE BANK)
# ==============================
def run_zkp_for_bank(bank_name, gradient):
    print(f"\n--- ZKP for {bank_name} ---")

    prover = ZKPProver(gradient)
    verifier = ZKPVerifier()

    # Step 1: Commit
    commitment = prover.commit()
    public_key = prover.get_public_key()
    print("  Commitment generated")

    # Step 2: Challenge
    challenge = verifier.generate_challenge()
    print(f"  Challenge received")

    # Step 3: Response
    response = prover.respond(challenge)
    print("  Response sent")

    # Step 4: Verify
    result = verifier.verify(commitment, response, public_key)

    if result:
        print(f"  ✅ ZKP PASSED — {bank_name}")
    else:
        print(f"  ❌ ZKP FAILED — {bank_name}")

    return result


# ==============================
# MAIN TEST
# ==============================
if __name__ == "__main__":
    print("===== Zero Knowledge Proof Tests =====")

    # Simulated gradients
    bank_gradients = {
        "HDFC": np.random.normal(0, 0.1, 100),
        "SBI": np.random.normal(0, 0.1, 100),
        "ICICI": np.random.normal(0, 0.1, 100),
        "Axis": np.random.normal(0, 0.1, 100),
        "GPay": np.random.normal(0, 0.1, 100),
        "PhonePe": np.random.normal(0, 0.1, 100),
    }

    results = {}

    # Run ZKP for all banks
    for bank, gradient in bank_gradients.items():
        results[bank] = run_zkp_for_bank(bank, gradient)

    # Summary
    print("\n===== ZKP Summary =====")
    for bank, passed in results.items():
        print(f"{bank:<10} {'✅ PASSED' if passed else '❌ FAILED'}")

    # ==============================
    # CHEATING TEST
    # ==============================
    print("\n===== Cheating Test =====")

    real_gradient = np.random.normal(0, 0.1, 100)

    prover = ZKPProver(real_gradient)
    verifier = ZKPVerifier()

    commitment = prover.commit()
    public_key = prover.get_public_key()
    challenge = verifier.generate_challenge()

    # Honest response
    response = prover.respond(challenge)
    result = verifier.verify(commitment, response, public_key)

    print(f"Honest proof: {'✅ PASSED' if result else '❌ FAILED'}")

    # Fake response (cheating)
    fake_response = secrets.randbelow(P - 1)
    result2 = verifier.verify(commitment, fake_response, public_key)

    print(f"Cheating attempt: {'❌ FAILED — Caught!' if not result2 else '⚠️ PASSED (Unexpected)'}")

    print("\n✅ ZKP module ready for integration!")