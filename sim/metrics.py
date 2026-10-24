import numpy as np

MEASURES = ["energy", "food_rate", "move_frac", "offspring_rate"]
METRIC_KEYS = (
    ["n_female", "n_male", "gini_all", "gini_female", "gini_male"]
    + [f"{m}_{k}" for m in MEASURES for k in ("gap", "p")]
)


def gini(values):
    v = np.sort(np.asarray(values, dtype=float))
    if len(v) == 0 or v.sum() <= 0:
        return 0.0
    cum = np.cumsum(v)
    return float((len(v) + 1 - 2 * cum.sum() / cum[-1]) / len(v))


def sex_gap(values, sexes):
    """Mean for males minus mean for females."""
    if (sexes == 0).sum() == 0 or (sexes == 1).sum() == 0:
        return 0.0
    return float(values[sexes == 1].mean() - values[sexes == 0].mean())


def permutation_p(values, sexes, rng, n=200):
    """How often do randomly shuffled labels give a gap this large?"""
    real = abs(sex_gap(values, sexes))
    hits = sum(abs(sex_gap(values, rng.permutation(sexes))) >= real
               for _ in range(n))
    return (hits + 1) / (n + 1)


def snapshot(agents, tick):
    rng = np.random.default_rng(tick)   # separate from the simulation's rng
    out = {k: 0.0 for k in METRIC_KEYS}
    out["n_female"] = sum(a.sex == 0 for a in agents)
    out["n_male"] = sum(a.sex == 1 for a in agents)

    energies = np.array([a.energy for a in agents])
    sexes_all = np.array([a.sex for a in agents])
    out["gini_all"] = gini(energies)
    out["gini_female"] = gini(energies[sexes_all == 0])
    out["gini_male"] = gini(energies[sexes_all == 1])

    adults = [a for a in agents if a.age > 0]
    if not adults:
        return out
    sexes = np.array([a.sex for a in adults])
    data = {
        "energy": np.array([a.energy for a in adults]),
        "food_rate": np.array([a.food_eaten / a.age for a in adults]),
        "move_frac": np.array([a.moves / a.age for a in adults]),
        "offspring_rate": np.array([a.offspring / a.age for a in adults]),
    }
    for m, vals in data.items():
        out[f"{m}_gap"] = sex_gap(vals, sexes)
        out[f"{m}_p"] = permutation_p(vals, sexes, rng)
    return out