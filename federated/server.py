import sys
from agents.llm_explainer import LLMExplainer

# ==============================
# 🔇 CLEAN OUTPUT
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

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("flwr").setLevel(logging.ERROR)
logging.getLogger("grpc").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# ==============================
# IMPORTS
# ==============================
import flwr as fl
import numpy as np

from privacy.zkp import ZKPVerifier
from agents.security_guard import SecurityGuard
from blockchain.web3_connector import log_round

# 🧠 RL MODULES
from agents.client_selector import ClientSelectorAgent
from agents.privacy_controller import PrivacyBudgetController


# ==============================
# 🔥 CUSTOM STRATEGY
# ==============================
class SecureFedAvg(fl.server.strategy.FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.guard = SecurityGuard(cosine_threshold=-0.3)
        self.explainer = LLMExplainer()

        self.clients_list = ["HDFC", "SBI", "ICICI", "Axis", "Kotak", "YesBank"]
        self.selector = ClientSelectorAgent(self.clients_list)
        self.privacy_controller = PrivacyBudgetController()

        self.prev_accuracy = 0.70

        # 🔥 BANK MAP
        self.bank_map = {
            "1": "HDFC",
            "2": "SBI",
            "3": "ICICI",
            "4": "Axis",
            "5": "Kotak",
            "6": "YesBank"
        }

    def aggregate_fit(self, rnd, results, failures):

        print("\n" + "=" * 50)
        print(f"🚀 ROUND {rnd}")
        print("=" * 50)

        verifier = ZKPVerifier()

        clean_results = []
        gradients_dict = {}
        blocked_banks = []

        # ==============================
        # STEP 1 — EXTRACT GRADIENTS
        # ==============================
        for client, fit_res in results:
            metrics = fit_res.metrics
            if not metrics:
                continue

            bank_id_num = str(metrics.get("bank_id", "unknown"))
            bank_id = self.bank_map.get(bank_id_num, "unknown")

            gradient = np.array([metrics.get("grad_norm", 0.0)])
            gradients_dict[bank_id] = gradient

        # ==============================
        # STEP 2 — VERIFY + SECURITY
        # ==============================
        for client, fit_res in results:
            metrics = fit_res.metrics
            if not metrics:
                continue

            bank_id_num = str(metrics.get("bank_id", "unknown"))
            bank_id = self.bank_map.get(bank_id_num, "unknown")

            commitment = int(metrics.get("zkp_commitment", 0))
            challenge = int(metrics.get("zkp_challenge", 0))
            response = int(metrics.get("zkp_response", 0))
            public_key = int(metrics.get("zkp_public_key", 0))

            # 🔐 ZKP
            is_valid = verifier.verify(
                commitment,
                response,
                public_key,
                challenge
            )

            if not is_valid:
                print(f"❌ Bank {bank_id} → ZKP FAILED")
                blocked_banks.append(bank_id)
                continue

            print(f"🏦 Bank {bank_id} → ZKP ✔")

            # 🛡 SECURITY
            gradient = gradients_dict[bank_id]

            is_clean, reason = self.guard.inspect(
                bank_id,
                gradient,
                gradients_dict
            )

            if is_clean:
                clean_results.append((client, fit_res))
            else:
                print(f"🚫 Bank {bank_id} BLOCKED → {reason}")
                blocked_banks.append(bank_id)

        # ==============================
        # STEP 3 — RL UPDATE
        # ==============================
        if len(clean_results) > 0:
            try:
                current_accuracy = clean_results[0][1].metrics.get(
                    "accuracy", self.prev_accuracy
                )
            except:
                current_accuracy = self.prev_accuracy

            improvement = current_accuracy - self.prev_accuracy

            for _, fit_res in clean_results:
                bank_id_num = str(fit_res.metrics.get("bank_id", "unknown"))
                bank_id = self.bank_map.get(bank_id_num, "unknown")

                self.selector.update(bank_id, improvement)

            print(f"\n📊 Accuracy: {current_accuracy:.4f}")
            self.selector.print_scores()

            epsilon = self.privacy_controller.adjust(
                current_accuracy, self.prev_accuracy
            )
            print(f"🔒 New epsilon: {epsilon:.2f}")

            self.prev_accuracy = current_accuracy

        # ==============================
        # SUMMARY
        # ==============================
        print("\n📊 Summary")
        print(f"   Clean Clients: {len(clean_results)} / {len(results)}")

        anomaly_detected = len(clean_results) < len(results)

        # ==============================
        # 🧠 LLM EXPLAINER
        # ==============================
        if anomaly_detected:
            print("\n🧠 AI FRAUD EXPLANATION:\n")

            transaction = {
                "type": "FL Update",
                "amount": 50000,
                "pattern": "Abnormal gradient detected",
            }

            explanation = self.explainer.explain(transaction)
            print(explanation)

        # ==============================
        # EMPTY CASE
        # ==============================
        if len(clean_results) == 0:
            print("\n⚠️ No valid clients → Skipping round")

            log_round(rnd, [], "no_model", True, "all_blocked")
            print("⛓️ Blockchain updated\n")

            return None, {}

        # ==============================
        # BLOCKCHAIN LOGGING
        # ==============================
        banks = [str(fit_res.metrics["bank_id"]) for _, fit_res in clean_results]

        anomaly_bank = ",".join(blocked_banks) if blocked_banks else "none"

        log_round(
            rnd,
            banks,
            "model_hash_placeholder",
            anomaly_detected,
            anomaly_bank
        )

        print("⛓️ Blockchain updated")

        # ==============================
        # AGGREGATION
        # ==============================
        print("✅ Aggregation complete")
        print("=" * 50)

        return super().aggregate_fit(rnd, clean_results, failures)


# ==============================
# ▶️ START SERVER
# ==============================
if __name__ == "__main__":
    print("\n🚀 Starting Secure FL Server...\n")

    strategy = SecureFedAvg(
        min_fit_clients=2,
        min_available_clients=2,
        min_evaluate_clients=1
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(
            num_rounds=10,
            round_timeout=60
        ),
        strategy=strategy,
    )