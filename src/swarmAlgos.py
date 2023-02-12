import random

from swarm import Swarm
from particle import Particle
import matplotlib.pyplot as plt
import numpy as np


# this class contains the main logic for the Particle Swarm Optimisation Algorithm
class ParticleSwarmAlgoSlow:
    def __init__(self, swarm_size, grid_size, target_position, v_min, v_max, w, c1, c2, iterations, ax):
        self.swarm_size = swarm_size
        self.grid_size = grid_size
        self.target_position = target_position
        self.v_min = v_min
        self.v_max = v_max
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.iterations = iterations
        self.ax = ax

        self.swarm = [Particle(grid_size, v_min, v_max) for _ in range(swarm_size)]
        self.global_best = Particle(grid_size, v_min, v_max)
        self.global_best.best_fitness = float('inf')
        self.fitness_scores = np.array([float('inf')] * swarm_size)

    def run(self):
        for i in range(self.iterations):

            self.show_current_state()
            plt.pause(0.1)

            for particle in self.swarm:

                self.show_current_state()
                plt.pause(10 ** -10)

                # updating velocities and positions
                self.update_velocity(particle)

                self.update_position(particle)

                curr_fitness = self.get_fitness(particle)

                if curr_fitness < particle.best_fitness:
                    particle.best_pos = particle.position
                    particle.best_fitness = curr_fitness
                    if curr_fitness < self.global_best.best_fitness:
                        self.global_best = particle

        plt.show()

    def update_position(self, particle):
        particle.position = particle.position + particle.velocity
        particle.position = np.clip(particle.position, 0, self.grid_size)

    def update_velocity(self, particle):
        particle.velocity = (
                self.w * particle.velocity +
                self.c1 * np.random.uniform(0, 1, 2) * (particle.best_pos - particle.position) +
                self.c2 * np.random.uniform(0, 1, 2) * (self.global_best.best_pos - particle.position)
        )
        particle.velocity = np.clip(particle.velocity, self.v_min, self.v_max)

    def show_current_state(self):
        self.ax.clear()
        x_coords = [p.position[0] for p in self.swarm]
        y_coords = [p.position[1] for p in self.swarm]
        self.ax.scatter(x_coords, y_coords, color='blue')
        self.ax.scatter(self.target_position[0], self.target_position[1], color='red', marker='x', s=100)
        self.ax.set_xlim(0, self.grid_size)
        self.ax.set_ylim(0, self.grid_size)

    def get_fitness(self, particle: Particle):
        return np.linalg.norm(particle.position - self.target_position)


class ParticleSwarmAlgoFast:
    def __init__(self, swarm_size, grid_size, target_position, v_min, v_max, w, c1, c2, iterations, ax):
        self.swarm_size = swarm_size
        self.grid_size = grid_size
        self.target_position = target_position
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.v_min = v_min
        self.v_max = v_max
        self.ax = ax
        self.iterations = iterations

        self.particle_positions = np.random.rand(swarm_size, 2) * grid_size
        self.particle_best_positions = self.particle_positions
        self.particle_velocities = np.random.rand(swarm_size, 2) * (self.v_max - self.v_min) + self.v_min
        self.particle_best_fitness = np.ones(swarm_size) * float('inf')
        self.global_best_position = self.particle_positions[np.argmin(self.particle_best_fitness)]
        self.particle_fitness = np.ones(swarm_size) * float('inf')

    def update_swarm(self):

        print(self.c1)
        print(np.random.rand(self.swarm_size, 2))
        print(self.particle_best_positions - self.particle_positions)

        self.particle_velocities = (
                self.w * self.particle_velocities +
                self.c1 * np.random.rand(self.swarm_size, 2) * (
                        self.particle_best_positions - self.particle_positions) +
                self.c2 * np.random.rand(self.swarm_size, 2) * (
                        self.global_best_position - self.particle_positions)
        )

        self.particle_velocities = np.clip(self.particle_velocities, self.v_min, self.v_max)

        self.particle_positions = np.clip(self.particle_positions + self.particle_velocities, 0, self.grid_size)

    def update_fitness(self):
        for i in range(len(self.particle_positions)):
            self.particle_fitness[i] = np.linalg.norm(self.particle_positions[i] - self.target_position)

        update_mask = self.particle_fitness < self.particle_best_fitness
        self.particle_best_positions[update_mask] = self.particle_positions[update_mask]
        # print(self.particle_best_fitness[update_mask])
        print(self.particle_fitness[update_mask])
        # self.particle_best_fitness[update_mask] = self.particle_fitness[update_mask]
        for i in range(len(update_mask)):
            if update_mask[i]:
                self.particle_best_fitness[i] = self.particle_fitness[i]
        self.global_best_position = self.particle_best_positions[np.argmin(self.particle_best_fitness)]

    def run(self):
        for i in range(self.iterations):
            self.update_swarm()
            self.update_fitness()

            # Clear the axis for the next iteration
            self.ax.clear()

            # Plot the swarm positions
            # print(self.swarm.swarm_positions)
            x_coords = [x for x, y in self.particle_positions]
            y_coords = [y for x, y in self.particle_positions]
            self.ax.scatter(x_coords, y_coords, color='blue')

            # Plot the target
            self.ax.scatter(self.target_position[0], self.target_position[1], color='red', marker='x', s=100)

            # Set the axis limits
            self.ax.set_xlim(0, self.grid_size)
            self.ax.set_ylim(0, self.grid_size)

            plt.pause(0.1)
        plt.show()


# this class contains the main logic for the Ant Colony Optimization Algorithm
#(self, swarm_size, grid_size, target_position, v_min, v_max, w, c1, c2, iterations, ax)
class AntOptimisationAlgo:
    def __init__(self, num_ants, grid_size, target_position, start, decay_rate, alpha, beta, iterations, q0, q, ax, maze):
        self.num_ants = num_ants
        self.grid_size = grid_size
        self.target_position = target_position
        self.start = start
        self.decay_rate = decay_rate
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.ax = ax
        self.q0 = q0
        self.q = q
        self.maze = maze

        print("hi")
        self.heuristics = np.ones((grid_size, grid_size))
        self.compute_heuristics()
        self.pheromones = np.ones((grid_size, grid_size)) / (1 / grid_size ** 2)
        self.probabilities = np.ones((grid_size, grid_size))
        self.compute_probabilities()


    def compute_heuristics(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                self.heuristics[i][j] = 1 / np.exp(1 + np.linalg.norm((i, j) - self.target_position))
                #self.heuristics[i][j] = 1 / 1 + np.linalg.norm((i, j) - self.target_position)

        self.heuristics = self.heuristics / self.heuristics.sum()

    def compute_probabilities(self):
        self.probabilities = np.power(self.pheromones, self.alpha) * np.power(self.heuristics, self.beta)
        self.probabilities = np.multiply(self.probabilities, self.maze.maze)
        self.probabilities = self.probabilities / self.probabilities.sum()


    def move_ant(self, ant):
        curr_pos = ant[-1]
        (x, y) = curr_pos
        next_move = curr_pos
        curr_prob = 0

        if random.random() < self.q0:
            # go to adjacent square with highest probability
            if (x - 1) in range(self.grid_size):
                if self.probabilities[x-1][y] > curr_prob:
                    curr_prob = self.probabilities[x-1][y]
                    next_move = (x-1, y)
            if (x + 1) in range(self.grid_size):
                if self.probabilities[x + 1][y] > curr_prob:
                    curr_prob = self.probabilities[x+1][y]
                    next_move = (x + 1, y)
            if (y - 1) in range(self.grid_size):
                if self.probabilities[x][y - 1] > curr_prob:
                    curr_prob = self.probabilities[x][y-1]
                    next_move = (x, y - 1)
            if (y + 1) in range(self.grid_size):
                if self.probabilities[x][y + 1] > curr_prob:
                    curr_prob = self.probabilities[x][y+1]
                    next_move = (x, y + 1)

        if next_move == curr_pos:
            # pick one of the adjacent squares with equal probability
            choices = []
            weights = []
            if (x - 1) in range(self.grid_size) and self.probabilities[x-1][y] != 0:
                choices.append((x - 1, y))
                weights.append(self.probabilities[x - 1][y])
            if (x + 1) in range(self.grid_size) and self.probabilities[x+1][y] != 0:
                choices.append((x + 1, y))
                weights.append(self.probabilities[x + 1][y])
            if (y - 1) in range(self.grid_size) and self.probabilities[x][y-1] != 0:
                choices.append((x, y - 1))
                weights.append(self.probabilities[x][y - 1])
            if (y + 1) in range(self.grid_size) and self.probabilities[x][y+1] != 0:
                choices.append((x, y + 1))
                weights.append(self.probabilities[x][y + 1])
            next_move = random.choices(choices, k=1)[0]

        if(self.probabilities[next_move[0]][next_move[1]] == 0):
            print("wrong!")
        print(next_move)
        print(self.maze.maze[next_move[0]][-next_move[1]])
        ant.append(next_move)
        return ant

    def update_pheromones(self, ants):
        self.pheromones = self.pheromones * self.decay_rate
        for ant in ants:
            for (i, j) in ant:
                self.pheromones[i][j] += self.q * 1 / len(ant)

        self.pheromones = self.pheromones / self.pheromones.sum()

    def run(self):
        self.ax.imshow(np.log(self.probabilities), cmap='hot')
        plt.pause(2)
        for i in range(self.iterations):
            ants = [[self.start] for _ in range(self.num_ants)]
            for ant in ants:
                count = 0
                while (ant[-1] != self.target_position).any():
                    if count > 100:
                        break
                    ant = self.move_ant(ant)

            self.update_pheromones(ants)
            self.compute_probabilities()

            self.ax.clear()


            for i, ant in enumerate(ants):
                antx = [p[1] for p in ant]
                anty = [p[0] for p in ant]

                label = 'ant' + str(i)
                self.ax.imshow(np.log(self.probabilities), cmap='hot', interpolation='nearest')
                self.ax.plot(antx, anty, color=np.random.rand(3, ), label = label)
                self.ax.legend()
                self.ax.scatter(self.target_position[0], self.target_position[1], color='red', marker='x', s=100)
            plt.pause(0.05)


        print('almost done')
        path_lengths = [len(ant) for ant in ants]
        best_ant = ants[np.argmin(path_lengths)]
        print(best_ant)
        antx = [p[1] for p in best_ant]
        anty = [p[0] for p in best_ant]
        self.ax.clear()
        self.ax.imshow(np.log(self.probabilities), cmap='hot', interpolation='nearest')
        self.ax.plot(antx, anty, color=np.random.rand(3, ), label='best ant')
        self.ax.legend()
        self.ax.scatter(self.target_position[0], self.target_position[1], color='red', marker='x', s=100)

        plt.savefig("output.png")
        plt.show()

        self.ax.imshow(np.log(self.probabilities.T + 10 ** (- 500)), cmap='hot', interpolation='nearest')
        self.ax.imshow(self.heuristics, cmap='hot', interpolation='nearest')
        self.ax.imshow(self.pheromones, cmap='hot', interpolation='nearest')
        plt.show()
        return ants



# this class contains the main logic for the Bee Algorithm
class BeeAlgo:

    def __init__(self):
        return
