import numpy as np
from particle import Particle


class Swarm:
    def __init__(self, swarm_size):
        self.swarm = [Particle() for _ in range(swarm_size)]
        self.size = swarm_size

    def initialize_swarm(self, grid_size):
        for i in range(self.size):
            self.swarm[i].particle_position = np.random.rand(1, 2) * grid_size
            self.swarm[i].particle_velocity = np.random.rand(1, 2) * 2 - 1
