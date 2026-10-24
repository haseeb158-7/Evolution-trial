import glob
import numpy as np
import pandas as pd

files = sorted(glob.glob("data/control/seed_*/log.csv"))
dfs = []
for f in files:
    d = pd.read_csv(f)
    d["seed"] = f
    dfs.append(d)
df = pd.concat(dfs)
late = df[df["tick"] > df["tick"].max() / 2]   # second half of each run

print(f"{len(files)} seeds, second half of each run\n")

ratio = (late["n_male"] / (late["n_male"] + late["n_female"])).mean()
print(f"mean male share of population: {ratio:.3f} (0.500 = equal)\n")

for m in ["energy", "food_rate", "move_frac", "offspring_rate"]:
    per_seed = late.groupby("seed")[m + "_gap"].mean()
    mean = per_seed.mean()
    ci = 1.96 * per_seed.std(ddof=1) / np.sqrt(len(per_seed))
    fp = (late[m + "_p"] < 0.05).mean()
    verdict = "includes 0" if abs(mean) <= ci else "EXCLUDES 0"
    print(f"{m}: mean gap {mean:.4f} +/- {ci:.4f}  [{verdict}]  "
          f"p<0.05 share {fp:.2f}")