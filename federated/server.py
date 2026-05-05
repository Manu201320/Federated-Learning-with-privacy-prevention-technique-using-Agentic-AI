import flwr as fl
import numpy as np

# 🔐 EXISTING MODULES
from privacy.zkp import ZKPVerifier
from agents.security_guard import SecurityGuard
from blockchain.web3_connector import log_round

# 🧠 YOUR MODULES
from agents.client_selector import ClientSelectorAgent
from agents.privacy_controller import PrivacyBudgetController


# ==============================
# 🔥 CUSTOM STRATEGY
# ==============================
class SecureFedAvg(fl.server.strategy.FedAvg):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.guard = SecurityGuard(cosine_threshold=-0.3)

        # 🧠 RL AGENTS
        self.clients_list = ["HDFC", "SBI", "ICICI", "Axis", "Kotak", "YesBank"]
        self.selector = ClientSelectorAgent(self.clients_list)
        self.privacy_controller = PrivacyBudgetController()

        self.prev_accuracy = 0.70

    def aggregate_fit(self, rnd, results, failures):
        print(f"\n🔐 Round {rnd} — Secure Aggregation\n")

        verifier = ZKPVerifier()

        clean_results = []
        gradients_dict = {}

        # Mapping (IMPORTANT FIX)
        bank_map = {
            "1": "HDFC",
            "2": "SBI",
            "3": "ICICI",
            "4": "Axis",
            "5": "Kotak",
            "6": "YesBank"
        }

        # ==============================
        # STEP 1 — EXTRACT GRADIENTS
        # ==============================
        for client, fit_res in results:
            metrics = fit_res.metrics
            if not metrics:
                continue

            bank_id_num = str(metrics.get("bank_id", "unknown"))
            bank_id = bank_map.get(bank_id_num, "unknown")

            gradient = np.array([metrics.get("grad_norm", 0.0)])
            gradients_dict[bank_id] = gradient

        # ==============================
        # STEP 2 — ZKP + SECURITY
        # ==============================
        for client, fit_res in results:
            metrics = fit_res.metrics
            if not metrics:
                continue

            bank_id_num = str(metrics.get("bank_id", "unknown"))
            bank_id = bank_map.get(bank_id_num, "unknown")

            commitment = int(metrics.get("zkp_commitment", 0))
            challenge = int(metrics.get("zkp_challenge", 0))
            response = int(metrics.get("zkp_response", 0))
            public_key = int(metrics.get("zkp_public_key", 0))

            is_valid = verifier.verify(commitment, response, public_key, challenge)

            if not is_valid:
                print(f"❌ ZKP FAILED — {bank_id}")
                continue

            print(f"✅ ZKP PASSED — {bank_id}")

            gradient = gradients_dict[bank_id]

            is_clean, reason = self.guard.inspect(
                bank_id,
                gradient,
                gradients_dict
            )

            if is_clean:
                clean_results.append((client, fit_res))
            else:
                print(f"🚫 BLOCKED — {bank_id} | {reason}")

        print(f"\n✅ Clean clients: {len(clean_results)} / {len(results)}")

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
                bank_id = bank_map.get(bank_id_num, "unknown")

                self.selector.update(bank_id, improvement)

            print(f"\n📊 Accuracy: {current_accuracy:.4f}")
            self.selector.print_scores()

            epsilon = self.privacy_controller.adjust(
                current_accuracy, self.prev_accuracy
            )
            print(f"🔒 New epsilon: {epsilon:.2f}")

            self.prev_accuracy = current_accuracy

        # ==============================
        # STEP 4 — EMPTY CASE
        # ==============================
        if len(clean_results) == 0:
            log_round(rnd, [], "no_model", True, "all_blocked")
            return None, {}

        # ==============================
        # STEP 5 — BLOCKCHAIN LOGGING
        # ==============================
        banks = [str(fit_res.metrics["bank_id"]) for _, fit_res in clean_results]

        anomaly_detected = len(clean_results) < len(results)
        anomaly_bank = "some_clients_blocked" if anomaly_detected else "none"

        log_round(
            rnd,
            banks,
            "model_hash_placeholder",
            anomaly_detected,
            anomaly_bank
        )

        # ==============================
        # STEP 6 — AGGREGATE
        # ==============================
        return super().aggregate_fit(rnd, clean_results, failures)


# ==============================
# ▶️ START SERVER
# ==============================
if __name__ == "__main__":
    print("🚀 Starting Secure FL Server with RL...")

    strategy = SecureFedAvg(
        min_fit_clients=1,
        min_available_clients=1,
        min_evaluate_clients=1
    )

    fl.server.start_server(
        server_address="localhost:8080",
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
    )