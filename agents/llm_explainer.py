import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# -----------------------------
# SAMPLE FRAUD CASES
# -----------------------------
cases = [
    "Large UPI transfer at 3am to new account",
    "Card used in two cities within 1 hour",
    "NEFT after new payee added at midnight",
    "Test ₹1 transaction followed by ₹50000 transfer",
]

# -----------------------------
# BUILD VECTOR DB
# -----------------------------
class FraudDB:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.data = cases

        embeddings = self.model.encode(self.data)

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype(np.float32))

        print("✅ DB Ready")

    def search(self, text):
        vec = self.model.encode([text]).astype(np.float32)
        _, idx = self.index.search(vec, 1)
        return self.data[idx[0][0]]


# -----------------------------
# EXPLAINER CLASS
# -----------------------------
class LLMExplainer:
    def __init__(self):
        self.db = FraudDB()

    def explain(self, transaction):
        similar = self.db.search(transaction["pattern"])

        explanation = f"""
🚨 FRAUD EXPLANATION

Transaction: ₹{transaction['amount']} via {transaction['type']}

Reason:
{transaction['pattern']}

Similar case:
{similar}

Action:
Block transaction and verify customer.
        """

        return explanation.strip()


# -----------------------------
# ✅ GLOBAL INSTANCE + FUNCTION (IMPORTANT FIX)
# -----------------------------
explainer = LLMExplainer()

def explain(description: str):
    """
    Wrapper for FastAPI
    """
    transaction = {
        "type": "UPI",
        "amount": 50000,
        "pattern": description
    }
    return explainer.explain(transaction)


# -----------------------------
# TEST
# -----------------------------
if __name__ == "__main__":
    print(explain("Large transfer at 2am to new account"))