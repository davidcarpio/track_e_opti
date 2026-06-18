## 2025-02-20 - [Vectorizing DP Optimization]
**Learning:** Python loops inside Dynamic Programming backward induction create massive bottlenecks in this specific codebase context. Due to grid evaluation dimensions `O(nodes * V^2)`, standard loop implementations become prohibitively expensive for realistic resolutions (100 nodes x 80 velocity levels).
**Action:** When working on grid-based calculations in DP optimization loops, always aggressively use vectorized operations via numpy matrix broadcasting to perform evaluation. Filtering calculations directly through numpy index masking yields two orders of magnitude in speedup while maintaining physical correctness.

## 2026-05-20 - [Track Grade & Spatial Resolution Constraints]
 **Learning:** [Raw GPS elevations contain high-frequency noise that produces wild gradients. Combining this with high spatial node counts (small `ds`) and fixed DP velocity grids causes massive acceleration penalties (dv^2/ds), forcing optimizers to artificially choose flat velocity profiles to avoid power spikes.]
 **Action:** [When processing raw GPS track data, always apply a smoothing filter (like a moving average) to elevations before calculating grades. In DP trajectory solvers, dynamically scale velocity grid resolution inversely with spatial step size (or proportional to `sqrt(num_nodes)`) to maintain reachable dv thresholds over small distances.]
## 2025-02-23 - [np.interp for Piecewise Vectorization]
**Learning:** Sequential boolean masking in NumPy (creating multiple `mask = ...` and assigning to `zeros_like`) generates massive temporary array allocations and bottlenecks hot loops like DP grid solvers.
**Action:** When calculating piecewise linear functions over large arrays, replace boolean masking with `np.interp` using precomputed static class arrays (`xp`, `yp`). This moves the branching logic entirely into compiled C code, speeding up the function by up to 3x.
## 2026-06-18 - [O(N) physics pre-calculation for DP grid]
**Learning:** In the Dynamic Programming optimizer, computing physics limits (like maximum traction and aerodynamic drag) directly on a broadcasted 2D (nv x nv) matrix results in O(N^2) redundant calculations. Because the maximum acceleration and deceleration limits from a specific state only depend on the starting velocity (`v_from`), they can be computed on a 1D array first.
**Action:** Always compute state-dependent physical limits on 1D arrays *before* broadcasting constraints to a 2D transition matrix to dramatically improve inner loop performance.
