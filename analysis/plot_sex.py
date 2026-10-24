import sys
import pandas as pd
import matplotlib.pyplot as plt

path = sys.argv[1] if len(sys.argv) > 1 else "data/run1/log.csv"
df = pd.read_csv(path)

fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 11))
ax[0].plot(df["tick"], df["n_female"], label="female")
ax[0].plot(df["tick"], df["n_male"], label="male")
ax[0].set_ylabel("number alive")
ax[0].legend()

ax[1].plot(df["tick"], df["energy_gap"])
ax[1].axhline(0, color="gray")
ax[1].set_ylabel("energy gap (M - F)")

ax[2].plot(df["tick"], df["gini_female"], label="female")
ax[2].plot(df["tick"], df["gini_male"], label="male")
ax[2].set_ylabel("energy inequality (Gini)")
ax[2].legend()

ax[3].plot(df["tick"], df["offspring_rate_gap"])
ax[3].axhline(0, color="gray")
ax[3].set_ylabel("offspring-rate gap (M - F)")
ax[3].set_xlabel("tick")

plt.tight_layout()
plt.savefig(path.replace("log.csv", "sex_overview.png"))

print("Share of log points with p < 0.05 (about 0.05 expected if no real gap):")
for m in ["energy", "food_rate", "move_frac", "offspring_rate"]:
    print(f"  {m}: {(df[m + '_p'] < 0.05).mean():.2f}")
plt.show()