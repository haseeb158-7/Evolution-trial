import argparse
import csv
import os
import pickle
import numpy as np

from sim.config import load_config
from sim.world import World
from sim.agent import Agent
from sim.evolve import reproduce
from sim.metrics import METRIC_KEYS, snapshot


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


def save_checkpoint(state, path):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)


def mean_or_nan(values):
    return float(np.mean(values)) if values else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--out", default="data/run1")
    args = parser.parse_args()

    cfg = load_config("config/base.yaml")
    if args.seed is not None:
        cfg["seed"] = args.seed
    a_cfg, run_cfg = cfg["agents"], cfg["run"]

    os.makedirs(args.out, exist_ok=True)
    ckpt = os.path.join(args.out, "checkpoint.pkl")
    log = os.path.join(args.out, "log.csv")

    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            state = pickle.load(f)
        print(f"Resuming from tick {state['tick']}")
    else:
        state = new_state(cfg)
        with open(log, "w", newline="") as f:
            csv.writer(f).writerow(
                ["tick", "population", "births", "rescues",
                 "lifespan_f", "lifespan_m"] + METRIC_KEYS)
        print(f"Starting a new run (seed {cfg['seed']})")

    rng, world = state["rng"], state["world"]
    agents, tick = state["agents"], state["tick"]
    births, rescues, dead = 0, 0, []

    while tick < run_cfg["total_ticks"]:
        world.step()
        for a in agents:
            a.step(world, a_cfg)

        room = a_cfg["max_population"] - len(agents)
        newborns = reproduce([a for a in agents if a.alive],
                             world.size, a_cfg, rng, room)
        births += len(newborns)

        dead += [(a.age, a.sex) for a in agents if not a.alive]
        agents = [a for a in agents if a.alive] + newborns

        while len(agents) < a_cfg["min_population"]:
            agents.append(random_agent(world, a_cfg, rng))
            rescues += 1

        tick += 1

        if tick % run_cfg["log_every"] == 0:
            snap = snapshot(agents, tick)
            lf = mean_or_nan([age for age, sx in dead if sx == 0])
            lm = mean_or_nan([age for age, sx in dead if sx == 1])
            with open(log, "a", newline="") as f:
                csv.writer(f).writerow(
                    [tick, len(agents), births, rescues,
                     round(lf, 2), round(lm, 2)]
                    + [round(snap[k], 4) for k in METRIC_KEYS])
            print(f"tick {tick}: pop={len(agents)} "
                  f"F={snap['n_female']} M={snap['n_male']} "
                  f"births={births} rescues={rescues}")
            births, rescues, dead = 0, 0, []

        if tick % run_cfg["checkpoint_every"] == 0:
            state.update(tick=tick, agents=agents)
            save_checkpoint(state, ckpt)

    state.update(tick=tick, agents=agents)
    save_checkpoint(state, ckpt)
    print("Finished.")


if __name__ == "__main__":
    main()