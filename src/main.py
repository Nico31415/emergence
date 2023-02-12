# # This is a sample Python script.
#
# # Press ⌃R to execute it or replace it with your code.
# # Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
#
import numpy as np
import matplotlib.pyplot as plt
import swarmAlgos
from maze import Maze
#hello

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    grid_size = 50
    target_position = np.random.random_integers(grid_size, size=2)
    target_position = np.array([grid_size - 1, grid_size - 1])

    swarm_size = 20
    iterations = 10

    c1 = 2.05
    c2 = 2.05
    w = 1

    v_min = -5
    v_max = 5

    fig, ax = plt.subplots()
    # fig2, ax2 = plt.subplots()

    pso = swarmAlgos.ParticleSwarmAlgoFast(swarm_size, grid_size, target_position, v_min, v_max, w, c1, c2, iterations,
                                           ax)
    start = [0, 0]
    decay_rate = 0.8
    alpha = 3
    beta = 6
    q0 = 0.4
    q = 3

    maze = Maze(grid_size, 0.2, start, target_position)
    print("hello")
    antAlgorithm = swarmAlgos.AntOptimisationAlgo(swarm_size, grid_size, target_position, (0, 0), decay_rate, alpha,
                                                  beta,
                                                  iterations, q0, q, ax, maze)

    antAlgorithm.run()
    # ax.clear()
    # pso.run()

# # See PyCharm help at https://www.jetbrains.com/help/pycharm/
