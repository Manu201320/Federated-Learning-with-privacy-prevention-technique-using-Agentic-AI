from web3 import Web3
import json
import hashlib
import numpy as np

# ==============================
# CONNECT TO LOCAL BLOCKCHAIN
# ==============================
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:8545"))

print("Connected to blockchain:", w3.is_connected())
print("Accounts available:", len(w3.eth.accounts))

# ✅ YOUR DEPLOYED CONTRACT ADDRESS
CONTRACT_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"

# Load ABI
with open("blockchain/artifacts/contracts/AuditTrail.sol/AuditTrail.json") as f:
    contract_json = json.load(f)
    ABI = contract_json["abi"]

contract = w3.eth.contract(address=CONTRACT_ADDRESS, abi=ABI)
account = w3.eth.accounts[0]


# ==============================
# HELPER — HASH MODEL WEIGHTS
# ==============================
def hash_model(weights_list):
    all_bytes = b""
    for w in weights_list:
        all_bytes += np.array(w).astype(np.float32).tobytes()
    return hashlib.sha256(all_bytes).hexdigest()


# ==============================
# LOG ROUND (FLEXIBLE VERSION)
# ==============================
def log_round(round_number, banks_selected, model_data,
              anomaly_detected=False, anomaly_bank=""):

    # 🔥 HANDLE BOTH CASES
    if isinstance(model_data, str):
        model_hash = model_data  # already hash/string
    else:
        model_hash = hash_model(model_data)  # compute hash

    tx = contract.functions.logRound(
        int(round_number),
        list(banks_selected),
        str(model_hash),
        bool(anomaly_detected),
        str(anomaly_bank)
    ).transact({"from": account})

    receipt = w3.eth.wait_for_transaction_receipt(tx)

    print(f"⛓️ Logged Round {round_number} on Blockchain")
    print(f"   Model hash: {model_hash[:20]}...")
    print(f"   TX hash:    {receipt.transactionHash.hex()[:20]}...")

    return model_hash


# ==============================
# UPDATE TRUST SCORE
# ==============================
def update_trust_score(bank_name, score_float):
    score_int = int(score_float * 100)

    tx = contract.functions.updateTrustScore(
        str(bank_name),
        int(score_int)
    ).transact({"from": account})

    w3.eth.wait_for_transaction_receipt(tx)

    print(f"✅ Trust score updated — {bank_name}: {score_float}")


# ==============================
# READ AUDIT LOG
# ==============================
def get_audit_log():
    count = contract.functions.getRoundsCount().call()

    print(f"\n===== Blockchain Audit Log ({count} rounds) =====\n")

    for i in range(count):
        round_num, model_hash, anomaly, anomaly_bank, timestamp = \
            contract.functions.getRound(i).call()

        print(f"Round {round_num}:")
        print(f"  Model Hash:  {model_hash[:20]}...")
        print(f"  Anomaly:     {'YES — ' + anomaly_bank if anomaly else 'No'}")
        print(f"  Timestamp:   {timestamp}")
        print()


# ==============================
# TEST
# ==============================
if __name__ == "__main__":
    print("\n🚀 Testing Blockchain Module\n")

    fake_weights = [np.random.normal(0, 0.1, 50) for _ in range(3)]

    # Works with weights
    log_round(1, ["HDFC", "SBI"], fake_weights)

    # Works with string (your server case)
    log_round(2, ["Axis", "GPay"], "model_hash_placeholder")

    update_trust_score("HDFC", 0.88)
    update_trust_score("ICICI", 0.55)

    get_audit_log()

    print("✅ Blockchain module complete!")