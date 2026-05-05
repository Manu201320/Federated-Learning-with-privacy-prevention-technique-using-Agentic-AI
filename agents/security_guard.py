import numpy as np
from scipy.spatial.distance import cosine


class SecurityGuard:
    def __init__(self, z_threshold=3.0, norm_threshold=3.0, cosine_threshold=-0.3):
        self.z_threshold = z_threshold
        self.norm_threshold = norm_threshold
        self.cosine_threshold = cosine_threshold
        self.blocked_history = []

    def inspect(self, client_name, gradient, all_gradients):
        """
        all_gradients = dict {client_name: gradient}
        Returns: (is_clean, reason)
        """

        other_gradients = [
            g for name, g in all_gradients.items() if name != client_name
        ]

        # ✅ Case 1: Only one client
        if len(other_gradients) == 0:
            return True, "Only client — auto accepted"

        # ✅ Case 2: Too few clients → skip strict checks
        if len(other_gradients) < 3:
            return True, "Few clients — relaxed security"

        other_array = np.array(other_gradients)

        # ==============================
        # CHECK 1 — NORM SIZE
        # ==============================
        this_norm = np.linalg.norm(gradient)
        norms = [np.linalg.norm(g) for g in other_gradients]
        avg_norm = np.mean(norms)

        if avg_norm < 1e-6:
            return True, "Norm too small — skip"

        if (this_norm / avg_norm) > self.norm_threshold:
            reason = f"Norm too large ({this_norm:.2f} vs avg {avg_norm:.2f})"
            self._block(client_name, reason)
            return False, reason

        # ==============================
        # CHECK 2 — Z-SCORE (SAFE)
        # ==============================
        mean_norm = np.mean(norms)
        std_norm = np.std(norms)

        # 🔥 FIX: avoid division explosion
        if std_norm < 1e-6:
            z_score = 0.0  # treat as normal
        else:
            z_score = abs(this_norm - mean_norm) / std_norm

        if z_score > self.z_threshold:
            reason = f"Z-score too high ({z_score:.2f})"
            self._block(client_name, reason)
            return False, reason

        # ==============================
        # CHECK 3 — COSINE SIMILARITY
        # ==============================
        avg_gradient = np.mean(other_array, axis=0)

        if np.linalg.norm(avg_gradient) < 1e-8 or np.linalg.norm(gradient) < 1e-8:
            cos_sim = 1.0
        else:
            cos_sim = 1 - cosine(gradient, avg_gradient)

        if np.isnan(cos_sim):
            cos_sim = 0.0

        if cos_sim < self.cosine_threshold:
            reason = f"Cosine similarity too low ({cos_sim:.2f})"
            self._block(client_name, reason)
            return False, reason

        return True, "Clean"

    def _block(self, client_name, reason):
        print(f"🚨 BLOCKED {client_name} — {reason}")
        self.blocked_history.append({
            "client": client_name,
            "reason": reason
        })

    def print_report(self):
        print("\n===== Security Guard Report =====")
        if not self.blocked_history:
            print("No attacks detected")
        else:
            for event in self.blocked_history:
                print(f"❌ {event['client']}: {event['reason']}")