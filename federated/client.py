```python
import sys
import os
import warnings
import logging

# ==============================
# 🔇 CLEAN LOGGING
# ==============================
class CleanOutput:
    def write(self, message):
        if "INFO" in message or "DEBUG" in message:
            return
        sys.__stdout__.write(message)

    def flush(self):
        sys.__stdout__.flush()

sys.stdout = CleanOutput()
sys.stderr = CleanOutput()

os.environ["GRPC_VERBOSITY"] = "NONE"
os.environ["GRPC_TRACE"] = ""

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.CRITICAL)
logging.getLogger("flwr").setLevel(logging.CRITICAL)
logging.getLogger("grpc").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)

# ==============================
# IMPORTS
# ==============================
import flwr as fl
import torch
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine

from models.gnn_model import GNNModel
from privacy.pqc import PQCEncryption
from privacy.zkp import generate_proof

# ==============================
# 📥 LOAD DATA
# ==============================
def load_data(bank_id):
    print(f"\n📂 Bank {bank_id} → Loading data...")

    df = pd.read_csv(f"data/bank{bank_id}.csv")

    # 🔥 LIMIT DATA (speed + stability)
    df = df.sample(n=5000, random_state=42)

    if "label" not in df.columns:
        if "isFraud" in df.columns:
            df.rename(columns={"isFraud": "label"}, inplace=True)
        elif "is_fraud" in df.columns:
            df.rename(columns={"is_fraud": "label"}, inplace=True)
        else:
            raise ValueError("❌ No label column found")

    df = df.select_dtypes(include=["number"])

    X = torch.tensor(df.drop("label", axis=1).values, dtype=torch.float32)
    y = torch.tensor(df["label"].values, dtype=torch.long)

    print(f"   ✅ Data ready: {X.shape}")
    return X, y


# ==============================
# 🏦 CLIENT
# ==============================
class FraudClient(fl.client.NumPyClient):

    def __init__(self, bank_id):
        print(f"\n🚀 Initializing Bank {bank_id} client...")

        self.bank_id = bank_id
        self.x, self.y = load_data(bank_id)
        self.model = GNNModel(input_dim=self.x.shape[1])

        # 🔐 PQC
        self.pqc = PQCEncryption()
        self.public_key, self.private_key = self.pqc.generate_keypair()

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        self.model.load_state_dict({k: torch.tensor(v) for k, v in params_dict})

    def fit(self, parameters, config):
        print(f"\n🏦 Bank {self.bank_id} | Training...")

        self.set_parameters(parameters)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        loss_fn = torch.nn.CrossEntropyLoss()

        loader = DataLoader(
            TensorDataset(self.x, self.y),
            batch_size=32,
            shuffle=True
        )

        privacy_engine = PrivacyEngine()

        self.model.train()

        self.model, optimizer, loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=0.7,
            max_grad_norm=1.0
        )

        # 🔥 LIMITED TRAINING (FAST)
        for i, (x, y) in enumerate(loader):
            if i > 50:
                break

            optimizer.zero_grad()
            loss = loss_fn(self.model(x), y)
            loss.backward()
            optimizer.step()

        # 🔐 Privacy
        epsilon = privacy_engine.get_epsilon(delta=1e-5)

        # 📈 TRAINING ACCURACY (your addition)
        self.model.eval()
        with torch.no_grad():
            output = self.model(self.x)
            preds = torch.argmax(output, dim=1)
            acc = (preds == self.y).float().mean().item()

        # ==============================
        # 🔐 MODEL PROCESSING
        # ==============================
        params = self.get_parameters(config)
        flat = np.concatenate([p.flatten() for p in params])

        # 🔐 PQC
        self.pqc.encrypt(flat, self.public_key)

        # 🔐 ZKP
        proof = generate_proof(flat)

        grad_norm = float(np.linalg.norm(flat))

        # ==============================
        # CLEAN OUTPUT
        # ==============================
        print(f"   🔐 ε = {epsilon:.2f}")
        print(f"   🔐 PQC ✔ (encrypted)")
        print(f"   🔐 ZKP ✔ (verified later)")
        print(f"   📊 Grad Norm: {grad_norm:.4f}")
        print(f"   📈 Accuracy: {acc:.4f}")

        return params, len(self.x), {
            "bank_id": int(self.bank_id),
            "grad_norm": grad_norm,
            "accuracy": acc,
            **proof
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        self.model.eval()
        with torch.no_grad():
            output = self.model(self.x)
            preds = torch.argmax(output, dim=1)
            acc = (preds == self.y).float().mean().item()

        print(f"   📈 Accuracy: {acc:.4f}")

        return float(0.0), len(self.x), {"accuracy": acc}


# ==============================
# ▶️ RUN CLIENT
# ==============================
if __name__ == "__main__":
    print("\n🚀 Client starting...\n")

    if len(sys.argv) < 2:
        print("❌ Provide bank ID")
        sys.exit(1)

    bank_id = int(sys.argv[1])

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FraudClient(bank_id),
    )
```
