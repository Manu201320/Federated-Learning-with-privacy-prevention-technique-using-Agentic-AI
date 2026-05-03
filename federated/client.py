import flwr as fl
import torch
import pandas as pd
import sys

from torch.utils.data import DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.metrics import precision_score, recall_score, f1_score

from models.gnn_model import GNNModel


# -----------------------------
# 📥 Load Data (FIXED + BALANCED)
# -----------------------------
def load_data(bank_id):
    print(f"📂 Loading data for bank {bank_id}...")

    df = pd.read_csv(f"data/bank{bank_id}.csv")

    # Fix label column
    if "label" not in df.columns:
        if "isFraud" in df.columns:
            df.rename(columns={"isFraud": "label"}, inplace=True)
        elif "is_fraud" in df.columns:
            df.rename(columns={"is_fraud": "label"}, inplace=True)
        else:
            raise ValueError("❌ No label column found")

    # 🔥 BALANCE DATA (MOST IMPORTANT FIX)
    fraud = df[df["label"] == 1]
    non_fraud = df[df["label"] == 0]

    # Prevent crash if very few fraud samples
    if len(fraud) > 0:
        non_fraud = non_fraud.sample(min(len(non_fraud), len(fraud) * 2))
        df = pd.concat([fraud, non_fraud]).sample(frac=1)

    # 🔧 Keep only useful numeric features
    df = df.select_dtypes(include=["number"])

    # Drop ID-like columns if present
    for col in ["sender_account", "receiver_account"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    X = df.drop("label", axis=1).values
    y = df["label"].values

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.long)

    print(f"✅ Balanced data: {X.shape}")

    return X, y


# -----------------------------
# 🏦 Client
# -----------------------------
class FraudClient(fl.client.NumPyClient):
    def __init__(self, bank_id):
        print("🚀 Initializing client...")
        self.x, self.y = load_data(bank_id)
        self.model = GNNModel(input_dim=self.x.shape[1])

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict)

    def fit(self, parameters, config):
        print("🏋️ Training started...")

        self.set_parameters(parameters)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        loss_fn = torch.nn.CrossEntropyLoss()

        dataset = TensorDataset(self.x, self.y)
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

        privacy_engine = PrivacyEngine()

        self.model.train()

        self.model, optimizer, data_loader = privacy_engine.make_private(
            module=self.model,
            optimizer=optimizer,
            data_loader=data_loader,
            noise_multiplier=0.7,   # ✅ balanced DP
            max_grad_norm=1.0
        )

        for batch_x, batch_y in data_loader:
            optimizer.zero_grad()
            output = self.model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimizer.step()

        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"🔐 Privacy ε: {epsilon:.2f}")

        return self.get_parameters(config), len(self.x), {}

    def evaluate(self, parameters, config):
        print("📊 Evaluating model...")

        self.set_parameters(parameters)

        self.model.eval()
        with torch.no_grad():
            output = self.model(self.x)
            preds = torch.argmax(output, dim=1)

            acc = (preds == self.y).float().mean().item()
            precision = precision_score(self.y.numpy(), preds.numpy(), zero_division=0)
            recall = recall_score(self.y.numpy(), preds.numpy(), zero_division=0)
            f1 = f1_score(self.y.numpy(), preds.numpy(), zero_division=0)

        print(f"📈 Accuracy: {acc:.4f}")
        print(f"🎯 Precision: {precision:.4f}")
        print(f"🔁 Recall: {recall:.4f}")
        print(f"🔥 F1 Score: {f1:.4f}")

        return float(0.0), len(self.x), {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


# -----------------------------
# ▶️ Run Client
# -----------------------------
if __name__ == "__main__":
    print("🚀 Client starting...")

    if len(sys.argv) < 2:
        print("❌ Provide bank ID")
        sys.exit(1)

    bank_id = int(sys.argv[1])

    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=FraudClient(bank_id),
    )