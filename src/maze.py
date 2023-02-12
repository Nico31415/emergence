import random

import matplotlib.pyplot as plt
import numpy as np


class Maze:
    def __init__(self, grid_size, wall_proportion, start, end):
        self.start = start
        self.end = end
        self.maze = np.ones((grid_size, grid_size))
        self.grid_size = grid_size
        self.wall_proportion = wall_proportion

        for i in range(grid_size):
            for j in range(grid_size):
                if random.random() < wall_proportion:
                    self.maze[i][j] = 0

        self.validate_maze()

    def validate_maze(self):
        stack = [self.start]
        self.maze[self.start[0]][self.start[1]] = 1
        visited = set()
        visited.add((self.start[0], self.start[1]))
        print(self.maze)

        while stack:
            print(stack)
            (x, y) = stack[-1]
            if ((x, y) == self.end).all():
                break
            choices = []
            good_choices = []
            if (x - 1) in range(self.grid_size) and ((x-1, y)) not in visited:
                if self.maze[x-1][y] == 1:
                    good_choices.append((x-1, y))
                choices.append((x-1, y))
            if (x + 1) in range(self.grid_size) and ((x+1, y)) not in visited:
                if self.maze[x+1][y] == 1:
                    good_choices.append((x+1, y))
                choices.append((x+1, y))
            if (y - 1) in range(self.grid_size) and ((x, y-1)) not in visited:
                if self.maze[x][y-1] == 1:
                    good_choices.append((x, y-1))
                choices.append((x, y-1))
            if (y + 1) in range(self.grid_size) and ((x, y+1)) not in visited:
                if self.maze[x][y+1] == 1:
                    good_choices.append((x, y+1))
                choices.append((x, y+1))

            if len(good_choices) > 0:
                next_pos = random.choice(good_choices)
                self.maze[next_pos[0]][next_pos[1]] = 1
                visited.add((next_pos[0], next_pos[1]))
                stack.append(next_pos)

            elif len(choices) > 0:
                next_pos = random.choice(choices)
                self.maze[next_pos[0]][next_pos[1]] = 1
                visited.add((next_pos[0], next_pos[1]))
                stack.append(next_pos)
            else:
                stack = stack[:-1]





    def show_maze(self, ax):
        ax.imshow(self.maze)
        plt.show()
