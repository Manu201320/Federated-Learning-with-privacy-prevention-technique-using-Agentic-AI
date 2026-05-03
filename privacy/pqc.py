import numpy as np

class PQCEncryption:
    def __init__(self):
        self.q = 3329

    def generate_keypair(self):
        private_key = np.random.randint(0, self.q, 256)
        public_key = private_key.copy()
        return public_key, private_key

    def encrypt(self, gradient, public_key):
        # Convert to integer
        grad_scaled = np.round(gradient * 1000).astype(int) % self.q

        # 🔥 FIX: match size
        pk = np.resize(public_key, len(grad_scaled))

        ciphertext = (grad_scaled + pk) % self.q
        return ciphertext

    def decrypt(self, ciphertext, private_key):
        sk = np.resize(private_key, len(ciphertext))

        decrypted = (ciphertext - sk) % self.q

        gradient = decrypted.astype(float) / 1000.0

        gradient[gradient > self.q / (2 * 1000)] -= self.q / 1000

        return gradient