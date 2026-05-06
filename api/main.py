from fastapi import FastAPI
import torch

from models.gnn_model import GNNModel
from agents.llm_explainer import explain

app = FastAPI()

# Load model (dummy global model)
from fastapi import FastAPI, Form
import torch

from agents.llm_explainer import explain

app = FastAPI()


# -----------------------------
# HOME
# -----------------------------
@app.get("/")
def home():
    return {"message": "Federated Fraud Detection API Running 🚀"}


# -----------------------------
# PREDICT
# -----------------------------
@app.post("/predict")
def predict(data: str = Form(...)):

    values = [float(x) for x in data.split(",")]

    # 🔥 SIMPLE DEMO LOGIC
    fraud = sum(values) > 5

    return {
        "fraud": fraud,
        "score": sum(values)
    }


# -----------------------------
# EXPLAIN
# -----------------------------
@app.post("/explain")
def explain_fraud(description: str = Form(...)):

    result = explain(description)

    return {
        "explanation": result
    }


# -----------------------------
# FULL ANALYSIS
# -----------------------------
@app.post("/analyze")
def analyze(
    description: str,
    data: str = Form(...)
):

    values = [float(x) for x in data.split(",")]

    # 🔥 Demo fraud logic
    fraud = sum(values) > 5

    explanation = ""

    if fraud:
        explanation = explain(description)

    return {
        "fraud": fraud,
        "score": sum(values),
        "explanation": explanation
    }# adjust if needed


@app.get("/")
def home():
    return {"message": "Federated Fraud Detection API Running 🚀"}


# -----------------------------
# 🔍 Fraud Prediction
# -----------------------------
@app.post("/predict")
def predict(data: list):
    x = torch.tensor([data], dtype=torch.float32)
    output = model(x)
    pred = torch.argmax(output, dim=1).item()

    return {
        "fraud": bool(pred),
        "prediction": int(pred)
    }


# -----------------------------
# 🧠 LLM Explanation
# -----------------------------
@app.post("/explain")
def explain_fraud(description: str):
    result = explain(description)
    return {"explanation": result}


# -----------------------------
# 🔥 FULL PIPELINE
# -----------------------------
@app.post("/analyze")
def analyze(data: list, description: str):

    x = torch.tensor([data], dtype=torch.float32)
    output = model(x)
    pred = torch.argmax(output, dim=1).item()

    explanation = ""
    if pred == 1:
        explanation = explain(description)

    return {
        "fraud": bool(pred),
        "explanation": explanation
    }