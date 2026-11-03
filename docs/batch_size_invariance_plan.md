# Additional Analysis Roadmap: Batch-Size Invariance of Online Ridge RLS

## 1. Theoretical Background & Paper Mandate

In Section 5.4 and Equation 10–13 of the paper:
**"F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning" (NeurIPS 2024)**,
the authors demonstrate that Recursive Least Squares (RLS) is an exact algebraic representation of batch ridge regression.

### 1.1 Mathematical Proof of Invariance
For a feature matrix $X \in \mathbb{R}^{N \times D}$ and one-hot target matrix $Y \in \mathbb{R}^{N \times K}$, the batch ridge solution with regularizer $\gamma$ is:
$$W^* = (X^T X + \gamma I)^{-1} X^T Y$$

Partition the dataset into arbitrary disjoint mini-batches $B_1, B_2, \dots, B_m$ such that $\sum_{k=1}^m |B_k| = N$:
$$X = \begin{bmatrix} X_1 \\ X_2 \\ \vdots \\ X_m \end{bmatrix}, \quad Y = \begin{bmatrix} Y_1 \\ Y_2 \\ \vdots \\ Y_m \end{bmatrix}$$

Then:
$$X^T X = \sum_{k=1}^m X_k^T X_k, \quad X^T Y = \sum_{k=1}^m X_k^T Y_k$$

By the Woodbury matrix identity (or sequential rank-$b$ updates), the sequence of recursive updates:
$$R_t = R_{t-1} - R_{t-1} X_t^T (I + X_t R_{t-1} X_t^T)^{-1} X_t R_{t-1}$$
$$W_t = W_{t-1} + R_t X_t^T (Y_t - X_t W_{t-1})$$
starting from $R_0 = \frac{1}{\gamma} I$ and $W_0 = 0$, satisfies at step $m$:
$$R_m = \left(\gamma I + \sum_{k=1}^m X_k^T X_k\right)^{-1} = (X^T X + \gamma I)^{-1}$$
$$W_m = R_m \left(\sum_{k=1}^m X_k^T Y_k\right) = W^*$$

**Conclusion**: The final model weights $W$ and covariance matrix $R$ are **strictly identical** regardless of batch size $B$, up to floating-point precision ($10^{-12}$ in `float64`).

---

## 2. Experimental Protocol (Additional Analysis 3A)

### 2.1 Evaluated Batch Sizes
We evaluate across 5 distinct batch sizes covering two orders of magnitude:
$$B \in \{5, 10, 25, 50, 100\}$$

### 2.2 Invariance Metrics
1. **Weight Deviation ($\Delta W$)**:
   $$\Delta W = \max_{j, k} |W_{B} - W_{B=10}|$$
2. **Accuracy Invariance ($\Delta \bar{A}$)**:
   $$\Delta \bar{A} = |\bar{A}_B - \bar{A}_{B=10}|$$
   Expected: $\Delta \bar{A} = 0.00\%$ across all $B$.
3. **Throughput vs Latency**:
   Measure training time and memory footprint as a function of $B$, validating that larger batch sizes reduce Woodbury matrix inversion overhead while producing identical accuracy.

---

## 3. Implementation Plan
- Standalone evaluation script: `scripts/analyze_batch_size_invariance.py`.
- Automated test: `tests/test_batch_size_invariance.py`.
- Integration into Final Reproduction Report.
