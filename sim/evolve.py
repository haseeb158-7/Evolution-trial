import numpy as np
from sim.agent import Agent

def try_reproduce(parent, world_size, cfg, rng):
    if parent.energy < cfg["repro_threshold"]:
        return None

    parent.energy -= cfg["start_energy"]   # child's energy comes from parent

    std = cfg["mutation_std"]
    w1 = parent.w1 + rng.normal(0, std, parent.w1.shape)
    w2 = parent.w2 + rng.normal(0, std, parent.w2.shape)

    x = int((parent.x + rng.integers(-1, 2)) % world_size)
    y = int((parent.y + rng.integers(-1, 2)) % world_size)
    return Agent(x, y, cfg, rng, weights=(w1, w2))