# Additional Analysis Roadmap: Batch-Size Invariance of Online Ridge RLS

## 1. Theoretical Background & Paper Mandate

In Section 5.4 and Equation 10–13 of the paper:
**"F-OAL: Forward-only Online Analytic Learning with Fast Training and Low Memory Footprint in Class Incremental Learning" (NeurIPS 2024)**,
the authors demonstrate that Recursive Least Squares (RLS) is an exact algebraic representation of batch ridge regression.

### 1.1 Exact Arithmetic vs Floating-Point Realities
In **exact infinite-precision arithmetic**, partition the dataset into arbitrary disjoint mini-batches $B_1, B_2, \dots, B_m$ such that $\sum_{k=1}^m |B_k| = N$:
$$X^T X = \sum_{k=1}^m X_k^T X_k, \quad X^T Y = \sum_{k=1}^m X_k^T Y_k$$

By the Woodbury matrix identity, the recursive updates:
$$R_t = R_{t-1} - R_{t-1} X_t^T (I + X_t R_{t-1} X_t^T)^{-1} X_t R_{t-1}$$
$$W_t = W_{t-1} + R_t X_t^T (Y_t - X_t W_{t-1})$$
starting from $R_0 = \frac{1}{\gamma} I$ and $W_0 = 0$, yield the exact batch ridge solution $W^* = (X^T X + \gamma I)^{-1} X^T Y$ at step $m$.

However, in **finite-precision computer arithmetic** (IEEE 754 `float64` or `float32`):
- Different batch sizes $B$ perform different sequences and numbers of matrix multiplications, inversions, and additions.
- The condition number $\kappa(I + X_t R_{t-1} X_t^T)$ varies with batch size.
- Accumulation of rounding errors means bitwise identity cannot be theoretically assumed a priori.

Therefore, this analysis does **not** assume beforehand that results are bitwise identical or that accuracy is invariant; instead, it establishes an empirical measurement protocol.

---

## 2. Experimental Measurement Protocol (Additional Analysis 3A)

### 2.1 Evaluated Batch Sizes
We evaluate across 5 distinct batch sizes covering two orders of magnitude:
$$B \in \{5, 10, 25, 50, 100\}$$
with fixed random seed (42), identical class order, and identical frozen feature representations.

### 2.2 Quantities to be Empirically Measured
Rather than predicting outcomes, the experiment will measure and report:

1. **Relative Weight Difference ($\epsilon_{W}^{(B)}$)**:
   $$\epsilon_{W}^{(B)} = \frac{\|W_B - W_{B=10}\|_F}{\|W_{B=10}\|_F}$$
   Measuring the Frobenius norm deviation of final weights relative to the baseline $B=10$.

2. **Closed-Form Ridge Residual ($\delta_{\text{closed}}^{(B)}$)**:
   $$\delta_{\text{closed}}^{(B)} = \frac{\|W_B - W_{\text{closed}}\|_F}{\|W_{\text{closed}}\|_F}$$
   Comparing sequential weights against the single-pass batch pseudoinverse solution.

3. **Empirical Accuracy Delta ($\Delta A_{avg}^{(B)}$ & $\Delta A_{last}^{(B)}$)**:
   $$\Delta A_{avg}^{(B)} = |A_{avg}^{(B)} - A_{avg}^{(B=10)}|$$
   Measuring whether floating-point weight discrepancies translate into any classification boundary flips.

4. **Numerical Condition Number ($\kappa(R)$)**:
   Tracking the spectral condition number of covariance matrix $R$ over tasks to monitor numerical stability.

5. **Execution Throughput & Runtime (Seconds)**:
   Measuring wall-clock time spent in Woodbury updates as a function of batch size $B$.

6. **Peak Memory Consumption (MB)**:
   Measuring memory allocation during the Woodbury matrix solve $(I_B + X_t R X_t^T)^{-1}$ which scales with $O(B^2)$.

---

## 3. Implementation Plan
- Standalone evaluation script: `scripts/analyze_batch_size_invariance.py`.
- Automated test: `tests/test_batch_size_invariance.py`.
- Empirical findings documented in final report.
