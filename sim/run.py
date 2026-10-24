import csv
import os
import pickle
import numpy as np

from sim.config import load_config
from sim.world import World
from sim.agent import Agent
from sim.evolve import try_reproduce

CONFIG_PATH = "config/base.yaml"
OUT_DIR = "data/run1"
CKPT = os.path.join(OUT_DIR, "checkpoint.pkl")
LOG = os.path.join(OUT_DIR, "log.csv")


def random_agent(world, a_cfg, rng):
    return Agent(int(rng.integers(world.size)),
                 int(rng.integers(world.size)), a_cfg, rng)


def new_state(cfg):
    rng = np.random.default_rng(cfg["seed"])
    world = World(cfg["world"], rng)
    a_cfg = cfg["agents"]
    agents = [random_agent(world, a_cfg, rng)
              for _ in range(a_cfg["start_population"])]
    return {"tick": 0, "rng": rng, "world": world, "agents": agents}


def save_checkpoint(state):
    tmp = CKPT + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, CKPT)   # safe: never leaves a half-written file


def main():
    cfg = load_config(CONFIG_PATH)
    a_cfg, run_cfg = cfg["agents"], cfg["run"]
    os.makedirs(OUT_DIR, exist_ok=True)

    if os.path.exists(CKPT):
        with open(CKPT, "rb") as f:
            state = pickle.load(f)
        print(f"Resuming from tick {state['tick']}")
    else:
        state = new_state(cfg)
        with open(LOG, "w", newline="") as f:
            csv.writer(f).writerow(
                ["tick", "population", "mean_energy",
                 "births", "mean_lifespan", "rescues"])
        print("Starting a new run")

    rng, world = state["rng"], state["world"]
    agents, tick = state["agents"], state["tick"]
    births, rescues, lifespans = 0, 0, []

    while tick < run_cfg["total_ticks"]:
        world.step()
        newborns = []
        for a in agents:
            a.step(world, a_cfg)
            if a.alive and len(agents) + len(newborns) < a_cfg["max_population"]:
                child = try_reproduce(a, world.size, a_cfg, rng)
                if child is not None:
                    newborns.append(child)
                    births += 1

        lifespans += [a.age for a in agents if not a.alive]
        agents = [a for a in agents if a.alive] + newborns

        # safety net: stop total extinction during early tuning
        while len(agents) < a_cfg["min_population"]:
            agents.append(random_agent(world, a_cfg, rng))
            rescues += 1

        tick += 1

        if tick % run_cfg["log_every"] == 0:
            mean_e = float(np.mean([a.energy for a in agents]))
            mean_l = float(np.mean(lifespans)) if lifespans else float("nan")
            with open(LOG, "a", newline="") as f:
                csv.writer(f).writerow(
                    [tick, len(agents), round(mean_e, 2),
                     births, round(mean_l, 2), rescues])
            print(f"tick {tick}: pop={len(agents)} lifespan={mean_l:.1f} "
                  f"births={births} rescues={rescues}")
            births, rescues, lifespans = 0, 0, []

        if tick % run_cfg["checkpoint_every"] == 0:
            state.update(tick=tick, agents=agents)
            save_checkpoint(state)

    state.update(tick=tick, agents=agents)
    save_checkpoint(state)
    print("Finished.")


if __name__ == "__main__":
    main()