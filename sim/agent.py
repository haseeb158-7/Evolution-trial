import numpy as np

# Actions: 0 stay, 1 up, 2 down, 3 left, 4 right
MOVES = [(0, 0), (0, -1), (0, 1), (-1, 0), (1, 0)]

class Agent:
    def __init__(self, x, y, cfg, rng, weights=None):
        self.x = x
        self.y = y
        self.energy = cfg["start_energy"]
        self.age = 0
        self.alive = True
        if weights is None:
            h = cfg["hidden_size"]
            self.w1 = rng.normal(0, 1, (6, h))   # 6 inputs
            self.w2 = rng.normal(0, 1, (h, 5))   # 5 actions
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

        # eat whatever food is on this cell
        self.energy += world.food[self.y, self.x]
        world.food[self.y, self.x] = 0.0

        self.age += 1
        if self.energy <= 0 or self.age >= cfg["max_age"]:
            self.alive = False