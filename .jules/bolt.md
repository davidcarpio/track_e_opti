## 2025-02-20 - [Vectorizing DP Optimization]
**Learning:** Python loops inside Dynamic Programming backward induction create massive bottlenecks in this specific codebase context. Due to grid evaluation dimensions `O(nodes * V^2)`, standard loop implementations become prohibitively expensive for realistic resolutions (100 nodes x 80 velocity levels).
**Action:** When working on grid-based calculations in DP optimization loops, always aggressively use vectorized operations via numpy matrix broadcasting to perform evaluation. Filtering calculations directly through numpy index masking yields two orders of magnitude in speedup while maintaining physical correctness.
## 2025-02-21 - [Vectorized numpy piecewise evaluation]
**Learning:** Using sequential boolean mask arrays (`np.zeros_like` + multiple `mask = condition; array[mask] = calc`) for piecewise functions inside heavy loops (like DP grid evaluations) causes massive performance overhead due to repeated array allocations and memory bandwidth limits.
**Action:** Replace sequential masking with `np.interp` when evaluating piecewise linear mappings on numpy arrays for a ~50% speedup.
