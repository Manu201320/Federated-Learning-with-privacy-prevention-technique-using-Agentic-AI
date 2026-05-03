import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("data/bank1.csv")

# Fix label column
if "is_fraud" in df.columns:
    df.rename(columns={"is_fraud": "label"}, inplace=True)
elif "isFraud" in df.columns:
    df.rename(columns={"isFraud": "label"}, inplace=True)

# Select numeric columns
numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c != "label"]

# ==============================
# SPLIT FEATURES (3 ORGS)
# ==============================
split = len(numeric_cols) // 3

bank_cols = numeric_cols[:split]
upi_cols = numeric_cols[split:2*split]
telecom_cols = numeric_cols[2*split:]

# Data
X_bank = df[bank_cols].fillna(0).values
X_upi = df[upi_cols].fillna(0).values
X_tel = df[telecom_cols].fillna(0).values
y = df["label"].values

# Normalize
X_bank = StandardScaler().fit_transform(X_bank)
X_upi = StandardScaler().fit_transform(X_upi)
X_tel = StandardScaler().fit_transform(X_tel)

# Convert to tensor
X_bank = torch.tensor(X_bank, dtype=torch.float32)
X_upi = torch.tensor(X_upi, dtype=torch.float32)
X_tel = torch.tensor(X_tel, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

print("Shapes:", X_bank.shape, X_upi.shape, X_tel.shape)

# ==============================
# LOCAL MODELS (per org)
# ==============================
class LocalModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 32)

    def forward(self, x):
        return torch.relu(self.fc(x))

# ==============================
# AGGREGATION MODEL
# ==============================
class Aggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(96, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, e1, e2, e3):
        x = torch.cat([e1, e2, e3], dim=1)
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

# ==============================
# INIT MODELS
# ==============================
model_bank = LocalModel(X_bank.shape[1])
model_upi = LocalModel(X_upi.shape[1])
model_tel = LocalModel(X_tel.shape[1])
agg = Aggregator()

# ==============================
# HANDLE CLASS IMBALANCE
# ==============================
class_counts = np.bincount(y.numpy())
weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = torch.optim.Adam(
    list(model_bank.parameters()) +
    list(model_upi.parameters()) +
    list(model_tel.parameters()) +
    list(agg.parameters()),
    lr=0.001
)

# ==============================
# TRAINING
# ==============================
epochs = 15

for epoch in range(epochs):
    optimizer.zero_grad()

    e1 = model_bank(X_bank)
    e2 = model_upi(X_upi)
    e3 = model_tel(X_tel)

    out = agg(e1, e2, e3)

    loss = criterion(out, y)
    loss.backward()
    optimizer.step()

    pred = torch.argmax(out, dim=1)
    acc = (pred == y).float().mean().item()

    print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Acc={acc:.4f}")

# ==============================
# EVALUATION
# ==============================
preds = torch.argmax(out, dim=1).numpy()
true = y.numpy()

print("\n===== CONFUSION MATRIX =====")
cm = confusion_matrix(true, preds)
print(cm)

print("\n===== CLASSIFICATION REPORT =====")
print(classification_report(true, preds, digits=4))

# ==============================
# SAVE MODEL
# ==============================
torch.save(agg.state_dict(), "models/vertical_model.pt")

print("\n✅ Vertical FL COMPLETE (WITH PROPER EVALUATION)!")