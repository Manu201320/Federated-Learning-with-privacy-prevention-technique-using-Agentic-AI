import numpy as np

# ==============================
# RL PRIVACY CONTROLLER
# ==============================
class PrivacyBudgetController:
    def __init__(self, epsilon_min=0.5, epsilon_max=3.0, initial_epsilon=1.5):
        self.epsilon = initial_epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_max = epsilon_max
        self.history = []

    def adjust(self, current_accuracy, prev_accuracy, attack_detected=False):
        """
        Policy:
        - Attack detected → increase privacy (lower epsilon)
        - Accuracy drops → reduce noise (increase epsilon)
        - Accuracy stable/improving → slightly increase privacy
        """

        if attack_detected:
            self.epsilon = max(self.epsilon_min, self.epsilon - 0.3)
            reason = "Attack detected → stronger privacy"

        elif current_accuracy < prev_accuracy - 0.02:
            self.epsilon = min(self.epsilon_max, self.epsilon + 0.2)
            reason = "Accuracy dropped → reduce noise"

        elif current_accuracy >= prev_accuracy:
            self.epsilon = max(self.epsilon_min, self.epsilon - 0.05)
            reason = "Stable accuracy → increase privacy"

        else:
            reason = "No change"

        self.history.append({
            "epsilon": self.epsilon,
            "accuracy": current_accuracy,
            "reason": reason
        })

        print(f"ε = {self.epsilon:.2f} | {reason}")
        return self.epsilon

    def print_history(self):
        print("\n===== PRIVACY HISTORY =====")
        for i, h in enumerate(self.history, 1):
            print(f"Round {i:02d} | ε={h['epsilon']:.2f} | acc={h['accuracy']:.4f} | {h['reason']}")


# ==============================
# TEST SIMULATION
# ==============================
if __name__ == "__main__":
    controller = PrivacyBudgetController()

    prev_acc = 0.70

    for round_num in range(1, 11):
        print(f"\n--- Round {round_num} ---")

        # simulate accuracy changes
        curr_acc = prev_acc + np.random.uniform(-0.01, 0.02)

        # simulate attack at round 5
        attack = (round_num == 5)

        controller.adjust(curr_acc, prev_acc, attack_detected=attack)

        prev_acc = curr_acc

    controller.print_history()

    print("\n✅ PRIVACY CONTROLLER COMPLETE")