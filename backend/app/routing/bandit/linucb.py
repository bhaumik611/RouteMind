import math
import threading
from typing import Dict, List, Optional, Tuple


class LinUCBArm:
    """Represents a single model arm in the Contextual Bandit space."""

    def __init__(self, arm_id: str, dimension: int = 10, alpha: float = 1.0):
        self.arm_id = arm_id
        self.d = dimension
        self.alpha = alpha  # Exploration parameter

        # A = I_d (Identity Matrix)
        self.A: List[List[float]] = [[1.0 if i == j else 0.0 for j in range(self.d)] for i in range(self.d)]
        # b = 0_d (Zero Vector)
        self.b: List[float] = [0.0] * self.d

        self.pulls: int = 0
        self.total_reward: float = 0.0
        self._cached_A_inv: Optional[List[List[float]]] = None

    def _invert_matrix_gauss_jordan(self, matrix: List[List[float]]) -> List[List[float]]:
        """Inverts a dxd positive-definite matrix using Gauss-Jordan elimination."""
        n = len(matrix)
        # Augment with identity
        aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matrix)]

        for i in range(n):
            # Pivot
            pivot = aug[i][i]
            if abs(pivot) < 1e-9:
                pivot = 1e-9
            for j in range(2 * n):
                aug[i][j] /= pivot

            for k in range(n):
                if k != i:
                    factor = aug[k][i]
                    for j in range(2 * n):
                        aug[k][j] -= factor * aug[i][j]

        return [row[n:] for row in aug]

    def get_A_inv(self) -> List[List[float]]:
        if self._cached_A_inv is None:
            self._cached_A_inv = self._invert_matrix_gauss_jordan(self.A)
        return self._cached_A_inv

    def compute_utility(self, x: List[float]) -> Tuple[float, float, float]:
        """Computes UCB score: p_a(x) = θ̂_a^T x + α * sqrt(x^T A_a^-1 x).
        Returns: (total_ucb_score, expected_reward, exploration_bonus)
        """
        A_inv = self.get_A_inv()

        # θ̂ = A^-1 b
        theta: List[float] = [0.0] * self.d
        for i in range(self.d):
            theta[i] = sum(A_inv[i][j] * self.b[j] for j in range(self.d))

        # Expected reward: θ̂^T x
        expected_reward = sum(theta[i] * x[i] for i in range(self.d))

        # Variance / uncertainty: x^T A^-1 x
        temp: List[float] = [0.0] * self.d
        for i in range(self.d):
            temp[i] = sum(x[j] * A_inv[j][i] for j in range(self.d))
        variance = sum(temp[i] * x[i] for i in range(self.d))
        variance = max(0.0, variance)

        exploration_bonus = self.alpha * math.sqrt(variance)
        ucb_score = expected_reward + exploration_bonus

        return round(ucb_score, 4), round(expected_reward, 4), round(exploration_bonus, 4)

    def update(self, x: List[float], reward: float) -> None:
        """Online Bayesian parameter update: A <- A + x x^T, b <- b + r x."""
        # Update A
        for i in range(self.d):
            for j in range(self.d):
                self.A[i][j] += x[i] * x[j]

        # Update b
        for i in range(self.d):
            self.b[i] += reward * x[i]

        self.pulls += 1
        self.total_reward += reward
        self._cached_A_inv = None  # Invalidate cached inverse


class DisjointLinUCB:
    """Multi-Arm Disjoint Linear Upper Confidence Bound Router."""

    def __init__(self, dimension: int = 10, alpha: float = 0.8):
        self.d = dimension
        self.alpha = alpha
        self.arms: Dict[str, LinUCBArm] = {}
        self._lock = threading.Lock()

    def _get_or_create_arm(self, arm_id: str) -> LinUCBArm:
        if arm_id not in self.arms:
            self.arms[arm_id] = LinUCBArm(arm_id=arm_id, dimension=self.d, alpha=self.alpha)
        return self.arms[arm_id]

    def select_arm(
        self, candidate_arm_ids: List[str], feature_vector: List[float]
    ) -> Tuple[str, float, float, float]:
        """Selects the arm with the highest UCB score from candidate list.
        Returns: (selected_arm_id, ucb_score, expected_reward, exploration_bonus)
        """
        with self._lock:
            best_arm_id = candidate_arm_ids[0]
            best_score = -float("inf")
            best_expected = 0.0
            best_bonus = 0.0

            for arm_id in candidate_arm_ids:
                arm = self._get_or_create_arm(arm_id)
                ucb, expected, bonus = arm.compute_utility(feature_vector)
                if ucb > best_score:
                    best_score = ucb
                    best_arm_id = arm_id
                    best_expected = expected
                    best_bonus = bonus

            return best_arm_id, best_score, best_expected, best_bonus

    def update_arm(self, arm_id: str, feature_vector: List[float], reward: float) -> None:
        """Applies online reward feedback to chosen arm."""
        with self._lock:
            arm = self._get_or_create_arm(arm_id)
            arm.update(feature_vector, reward)
