import numpy as np

# Actions: 0 stay, 1 up, 2 down, 3 left, 4 right
MOVES = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]
FEMALE, MALE = 0, 1


class Agent:
    def __init__(self, x, y, cfg, rng, weights=None, sex=None):
        self.x = x
        self.y = y
        self.energy = cfg["start_energy"]
        self.age = 0
        self.alive = True
        self.sex = int(rng.integers(2)) if sex is None else sex
        self.food_eaten = 0.0
        self.moves = 0
        self.offspring = 0
        if weights is None:
            h = cfg["hidden_size"]
            self.w1 = rng.normal(0, 1, (6, h))
            self.w2 = rng.normal(0, 1, (h, 5))
        else:
            self.w1, self.w2 = weights

    def sense(self, world, cfg):
        s, x, y, f = world.size, self.x, self.y, world.food
        return np.array([
            f[y, x],
            f[(y - 1) % s, x],
            f[(y + 1) % s, x],
            f[y, (x - 1) % s],
            f[y, (x + 1) % s],
            self.energy / cfg["start_energy"],
        ]) / np.array([world.food_max] * 5 + [1.0])

    def step(self, world, cfg):
        inputs = self.sense(world, cfg)
        hidden = np.tanh(inputs @ self.w1)
        action = int(np.argmax(hidden @ self.w2))

        dx, dy = MOVES[action]
        self.x = (self.x + dx) % world.size
        self.y = (self.y + dy) % world.size

        self.energy -= cfg["base_cost"]
        if action != 0:
            self.energy -= cfg["move_cost"]
            self.moves += 1

        eaten = float(world.food[self.y, self.x])
        self.energy += eaten
        self.food_eaten += eaten
        world.food[self.y, self.x] = 0.0

        self.age += 1
        if self.energy <= 0 or self.age >= cfg["max_age"]:
            self.alive = False