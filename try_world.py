import numpy as np
from sim.config import load_config
from sim.world import World
from sim.agent import Agent

cfg = load_config("config/base.yaml")
rng = np.random.default_rng(cfg["seed"])

world = World(cfg["world"], rng)
a_cfg = cfg["agents"]
agents = [
    Agent(rng.integers(world.size), rng.integers(world.size), a_cfg, rng)
    for _ in range(a_cfg["start_population"])
]

for tick in range(300):
    world.step()
    for a in agents:
        a.step(world, a_cfg)
    agents = [a for a in agents if a.alive]
    if tick % 50 == 0:
        avg = np.mean([a.energy for a in agents]) if agents else 0
        print(f"tick {tick}: alive={len(agents)}, avg energy={avg:.1f}")