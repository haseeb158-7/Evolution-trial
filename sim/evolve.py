import numpy as np
from sim.agent import Agent, FEMALE, MALE


def reproduce(agents, world_size, cfg, rng, max_new):
    """Pair eligible females with nearby eligible males."""
    if max_new <= 0:
        return []

    eligible = [a for a in agents if a.energy >= cfg["repro_threshold"]]
    cells = {}
    for a in eligible:
        cells.setdefault((a.x, a.y), []).append(a)

    females = [a for a in eligible if a.sex == FEMALE]
    rng.shuffle(females)

    used = set()
    newborns = []
    s = world_size
    std = cfg["mutation_std"]
    cost = cfg["start_energy"] / 2

    for mom in females:
        if len(newborns) >= max_new:
            break
        if id(mom) in used:
            continue

        candidates = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = ((mom.x + dx) % s, (mom.y + dy) % s)
                for other in cells.get(key, []):
                    if other.sex == MALE and id(other) not in used:
                        candidates.append(other)
        if not candidates:
            continue

        dad = candidates[int(rng.integers(len(candidates)))]
        used.add(id(mom))
        used.add(id(dad))

        mom.energy -= cost
        dad.energy -= cost
        mom.offspring += 1
        dad.offspring += 1

        # child brain: each weight comes from mom or dad, then mutates
        m1 = rng.random(mom.w1.shape) < 0.5
        m2 = rng.random(mom.w2.shape) < 0.5
        w1 = np.where(m1, mom.w1, dad.w1) + rng.normal(0, std, mom.w1.shape)
        w2 = np.where(m2, mom.w2, dad.w2) + rng.normal(0, std, mom.w2.shape)

        newborns.append(Agent(mom.x, mom.y, cfg, rng, weights=(w1, w2)))

    return newborns