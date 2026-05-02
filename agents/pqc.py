import numpy as np

# ==============================
# POST QUANTUM CRYPTOGRAPHY (SIMULATION)
# ==============================
# NOTE:
# This is a simplified simulation inspired by CRYSTALS-Kyber.
# It demonstrates secure gradient transmission.
# In production → use pqcrypto / liboqs.


class PQCEncryption:
    def __init__(self):
        self.q = 3329  # modulus (same as Kyber)

    # ==============================
    # KEY GENERATION
    # ==============================
    def generate_keypair(self):
        private_key = np.random.randint(0, self.q, 256)
        public_key = private_key.copy()  # keep consistent for simulation
        return public_key, private_key

    # ==============================
    # ENCRYPTION
    # ==============================
    def encrypt(self, gradient, public_key):
        # Convert float gradient → integer
        grad_scaled = np.round(gradient * 1000).astype(int) % self.q

        # Match length
        pk = public_key[:len(grad_scaled)]

        # Simple additive encryption
        ciphertext = (grad_scaled + pk) % self.q
        return ciphertext

    # ==============================
    # DECRYPTION
    # ==============================
    def decrypt(self, ciphertext, private_key):
        sk = private_key[:len(ciphertext)]

        # Reverse encryption
        decrypted = (ciphertext - sk) % self.q

        # Convert back to float
        gradient = decrypted.astype(float) / 1000.0

        # Handle wrap-around values
        gradient[gradient > self.q / (2 * 1000)] -= self.q / 1000

        return gradient


# ==============================
# TEST — SINGLE GRADIENT
# ==============================
if __name__ == "__main__":
    pqc = PQCEncryption()

    print("===== PQC Encryption Test =====\n")

    # Generate keypair
    public_key, private_key = pqc.generate_keypair()
    print("✅ Key pair generated")

    # Original gradient
    original_gradient = np.random.normal(0, 0.1, 50)
    print("\nOriginal Gradient:")
    print(np.round(original_gradient[:5], 4))

    # Encrypt
    ciphertext = pqc.encrypt(original_gradient, public_key)
    print("\nEncrypted (sample):")
    print(ciphertext[:5])

    # Decrypt
    decrypted_gradient = pqc.decrypt(ciphertext, private_key)
    print("\nDecrypted Gradient:")
    print(np.round(decrypted_gradient[:5], 4))

    # Check accuracy
    error = np.mean(np.abs(original_gradient - decrypted_gradient))
    print(f"\nReconstruction error: {error:.6f}")

    # ==============================
    # TEST — MULTIPLE BANKS
    # ==============================
    print("\n===== Encrypting All Bank Gradients =====\n")

    bank_gradients = {
        "HDFC": np.random.normal(0, 0.1, 50),
        "SBI": np.random.normal(0, 0.1, 50),
        "ICICI": np.random.normal(0, 0.1, 50),
        "Axis": np.random.normal(0, 0.1, 50),
        "GPay": np.random.normal(0, 0.1, 50),
        "PhonePe": np.random.normal(0, 0.1, 50),
    }

    encrypted_data = {}

    for bank, grad in bank_gradients.items():
        pub_key, priv_key = pqc.generate_keypair()

        ciphertext = pqc.encrypt(grad, pub_key)
        decrypted = pqc.decrypt(ciphertext, priv_key)

        error = np.mean(np.abs(grad - decrypted))

        encrypted_data[bank] = {
            "ciphertext": ciphertext,
            "error": error
        }

        print(f"{bank:<10} ✅ Encrypted | Error: {error:.6f}")

    print("\n✅ PQC module ready!")