import numpy as np

class Particle:
    def __init__(self, grid_size, v_min, v_max):
        self.position = np.random.uniform(0, grid_size, 2)
        self.position = np.random.uniform(0, grid_size, 2)
        self.velocity = np.random.uniform(v_min, v_max, 2)
        self.best_pos = self.position
        self.best_fitness = float('inf')

        #print(self.best_pos)
