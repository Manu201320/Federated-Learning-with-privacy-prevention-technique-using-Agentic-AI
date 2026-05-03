import numpy as np

# ==============================
# RL CLIENT SELECTOR (UCB1)
# ==============================
class ClientSelectorAgent:
    def __init__(self, clients):
        self.clients = clients
        self.n = len(clients)

        self.counts = np.zeros(self.n)
        self.rewards = np.zeros(self.n)
        self.trust_scores = {c: 0.75 for c in clients}

    def select_clients(self, n_select=4):
        total = np.sum(self.counts) + 1
        scores = []

        for i in range(self.n):
            if self.counts[i] == 0:
                scores.append(float('inf'))  # explore
            else:
                avg = self.rewards[i]
                conf = np.sqrt(2 * np.log(total) / self.counts[i])
                scores.append(avg + conf)

        selected_idx = np.argsort(scores)[-n_select:]
        return [self.clients[i] for i in selected_idx]

    def update(self, client_name, improvement):
        idx = self.clients.index(client_name)
        self.counts[idx] += 1
        n = self.counts[idx]

        # update reward (running average)
        self.rewards[idx] = ((n - 1) * self.rewards[idx] + improvement) / n

        # update trust score
        if improvement > 0:
            self.trust_scores[client_name] = min(1.0, self.trust_scores[client_name] + 0.03)
        else:
            self.trust_scores[client_name] = max(0.0, self.trust_scores[client_name] - 0.07)

    def print_scores(self):
        print("\n===== TRUST SCORES =====")
        for c, s in self.trust_scores.items():
            bar = "█" * int(s * 20)
            print(f"{c:<10} {s:.2f} {bar}")


# ==============================
# TEST WITH REALISTIC DIFFERENCES
# ==============================
if __name__ == "__main__":
    clients = ["HDFC", "SBI", "ICICI", "Axis", "Kotak", "YesBank"]
    agent = ClientSelectorAgent(clients)

    # Simulated client quality (IMPORTANT)
    client_quality = {
        "HDFC": 0.020,     # best
        "SBI": 0.018,
        "ICICI": 0.014,
        "Axis": 0.010,
        "Kotak": 0.005,
        "YesBank": -0.005  # bad client
    }

    prev_acc = 0.70

    for round_num in range(1, 11):
        print(f"\n--- Round {round_num} ---")

        selected = agent.select_clients()
        print("Selected:", selected)

        # Calculate improvement based on selected clients
        improvements = []
        for c in selected:
            base = client_quality[c]
            noise = np.random.uniform(-0.002, 0.002)
            improvements.append(base + noise)

        avg_improvement = np.mean(improvements)
        new_acc = prev_acc + avg_improvement

        # Update agent individually per client
        for c, imp in zip(selected, improvements):
            agent.update(c, imp)

        prev_acc = new_acc

        print(f"Accuracy: {new_acc:.4f}")
        agent.print_scores()

    print("\n✅ RL CLIENT SELECTOR COMPLETE (SMART VERSION)")