import numpy as np

class World:
    def __init__(self, cfg, rng):
        self.size = cfg["size"]
        self.food_max = cfg["food_max"]
        self.food_regrow = cfg["food_regrow"]
        # True on cells where food can grow
        self.fertile = rng.random((self.size, self.size)) < cfg["fertile_fraction"]
        self.food = np.where(self.fertile, self.food_max, 0.0)

    def step(self):
        grown = self.food + self.food_regrow
        grown = np.minimum(grown, self.food_max)
        self.food = np.where(self.fertile, grown, 0.0)
