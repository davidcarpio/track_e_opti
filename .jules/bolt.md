## 2025-02-20 - [Vectorizing DP Optimization]
**Learning:** Python loops inside Dynamic Programming backward induction create massive bottlenecks in this specific codebase context. Due to grid evaluation dimensions `O(nodes * V^2)`, standard loop implementations become prohibitively expensive for realistic resolutions (100 nodes x 80 velocity levels).
**Action:** When working on grid-based calculations in DP optimization loops, always aggressively use vectorized operations via numpy matrix broadcasting to perform evaluation. Filtering calculations directly through numpy index masking yields two orders of magnitude in speedup while maintaining physical correctness.
## 2025-06-05 - [Optimize Numpy Boolean Masking for Piecewise Functions]
**Learning:** Using multiple sequential boolean masks for piecewise functions on large numpy arrays (e.g. mapping motor efficiency) is significantly slower (by ~60%) than using `np.interp` paired with `np.where` for constant edge cases.
**Action:** When implementing or optimizing 1D piecewise logic or lookup tables that process large numerical arrays, prefer `np.interp` over explicit manual branching with boolean arrays.
