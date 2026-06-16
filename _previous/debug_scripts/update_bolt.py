import os
import datetime

bolt_file = ".jules/bolt.md"
date_str = datetime.datetime.now().strftime("%Y-%m-%d")

content = f"""## {date_str} - [Track Grade & Spatial Resolution Constraints]
 **Learning:** [Raw GPS elevations contain high-frequency noise that produces wild gradients. Combining this with high spatial node counts (small `ds`) and fixed DP velocity grids causes massive acceleration penalties (dv^2/ds), forcing optimizers to artificially choose flat velocity profiles to avoid power spikes.]
 **Action:** [When processing raw GPS track data, always apply a smoothing filter (like a moving average) to elevations before calculating grades. In DP trajectory solvers, dynamically scale velocity grid resolution inversely with spatial step size (or proportional to `sqrt(num_nodes)`) to maintain reachable dv thresholds over small distances.]"""

os.makedirs(".jules", exist_ok=True)
with open(bolt_file, "a") as f:
    f.write("\n\n" + content + "\n")
