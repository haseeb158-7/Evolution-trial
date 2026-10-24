import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/run1/log.csv")

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 9))
axes[0].plot(df["tick"], df["population"])
axes[0].set_ylabel("population")
axes[1].plot(df["tick"], df["mean_lifespan"])
axes[1].set_ylabel("mean lifespan of dead")
axes[2].plot(df["tick"], df["rescues"])
axes[2].set_ylabel("random rescues")
axes[2].set_xlabel("tick")
plt.tight_layout()
plt.savefig("data/run1/overview.png")
plt.show()