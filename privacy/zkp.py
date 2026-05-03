import numpy as np
import hashlib
import secrets

# ==============================
# SCHNORR ZKP PARAMETERS
# ==============================

P = 10**9 + 7   # smaller prime (fits in uint64)   # large prime
G = 2


# ==============================
# PROVER (CLIENT)
# ==============================
class ZKPProver:
    def __init__(self, gradient):
        # gradient → hash → integer secret
        gradient_bytes = gradient.astype(np.float32).tobytes()
        hash_hex = hashlib.sha256(gradient_bytes).hexdigest()
        self.secret = int(hash_hex, 16) % P

    def commit(self):
        self.nonce = secrets.randbelow(P - 1)
        self.commitment = pow(G, self.nonce, P)
        return int(self.commitment)

    def respond(self, challenge):
        return int((self.nonce + challenge * self.secret) % (P - 1))

    def get_public_key(self):
        return int(pow(G, self.secret, P))


# ==============================
# VERIFIER (SERVER)
# ==============================
class ZKPVerifier:
    def generate_challenge(self):
        return int(secrets.randbelow(P - 1))

    def verify(self, commitment, response, public_key, challenge):
        # Ensure all integers (CRITICAL FIX)
        commitment = int(commitment)
        response = int(response)
        public_key = int(public_key)
        challenge = int(challenge)

        lhs = pow(G, response, P)
        rhs = (commitment * pow(public_key, challenge, P)) % P

        return lhs == rhs


# ==============================
# HELPER (FOR CLIENT INTEGRATION)
# ==============================
def generate_proof(gradient):
    prover = ZKPProver(gradient)

    commitment = prover.commit()
    public_key = prover.get_public_key()

    # challenge generated on server normally,
    # but for Flower we simulate here
    challenge = secrets.randbelow(P - 1)

    response = prover.respond(challenge)

    return {
        "zkp_commitment": int(commitment),
        "zkp_challenge": int(challenge),
        "zkp_response": int(response),
        "zkp_public_key": int(public_key),
    }


# ==============================
# TEST
# ==============================
if __name__ == "__main__":
    print("===== ZKP TEST =====")

    gradient = np.random.normal(0, 0.1, 100)

    proof = generate_proof(gradient)

    verifier = ZKPVerifier()

    result = verifier.verify(
        proof["zkp_commitment"],
        proof["zkp_response"],
        proof["zkp_public_key"],
        proof["zkp_challenge"]
    )

    print("ZKP Result:", "✅ PASSED" if result else "❌ FAILED")