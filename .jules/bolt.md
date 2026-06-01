## 2025-06-01 - [NumPy Vectorization Optimization]
**Learning:** Sequential boolean masking in NumPy is a significant performance anti-pattern. Functions evaluated across the entire array (`np.where` or `np.interp`) are inherently faster because they leverage underlying C code directly without the overhead of computing subsets and dynamically allocating matching masks iteratively.
**Action:** Always prefer `np.interp` over multiple boolean statements for any piece-wise logic scaling. Calculate inputs (like absolute values) beforehand across the entire array.
