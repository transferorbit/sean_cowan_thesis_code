"""HIMMELBLAU FUNCTION OPTIMIZATION USING PYGMO"""

import pygmo as pg
import math
import numpy as np
import matplotlib.pyplot as plt

class HimmelblauOptimization:
    """
    This class defines a PyGMO-compatible User-Defined Optimization Problem.
    """

    def __init__(self,
                 x_min: float,
                 x_max: float,
                 y_min: float,
                 y_max: float):
        """
        Constructor for the HimmelblauOptimization class.
        """
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max

    def get_bounds(self):
        """
        Defines the boundaries of the search space.
        """
        return ([self.x_min, self.y_min], [self.x_max, self.y_max])

    def fitness(self,
                x: list):
        """
        Computes the fitness value for the problem.
        """
        function_value = math.pow(x[0] * x[0] + x[1] - 11.0, 2.0) + math.pow(x[0] + x[1] * x[1] - 7.0, 2.0)
        return [function_value]

# Instantiation of the UDP problem
udp = HimmelblauOptimization(-5.0, 5.0, -5.0, 5.0)

# Creation of the pygmo problem object
prob = pg.problem(udp)

# Define number of generations
number_of_generations = 1
# Create Differential Evolution object by passing the number of generations as input
algo = pg.algorithm(pg.gaco(gen=number_of_generations))

# Set population size
pop_size = 1000
# Set seed
current_seed = 171015
# Create population
pop = pg.population(prob, size=pop_size, seed=current_seed)

# Set number of evolutions
number_of_evolutions = 100
# Initialize empty containers
individuals_list = []
fitness_list = []

# plot
font_size = 20
plt.rcParams.update({'font.size': font_size})
plt.figure(figsize=(16, 12))
plt.grid()

# Evolve population multiple times
for i in range(number_of_evolutions):
    print("Current evolution: " + str(i))
    pop = algo.evolve(pop)
    individuals_list.append(pop.get_x()[pop.best_idx()])
    fitness_list.append(pop.get_f()[pop.best_idx()])
    plt.scatter(pop.champion_x[0], pop.champion_x[1], marker='x', color='black', s=100)
    print(pop.champion_f)
    print(pop.champion_x)

print("Population champion: ", pop.champion_x)

# def himmelblaufunct(x):
#     function_value = math.pow(x[0] * x[0] + x[1] - 11.0, 2.0) + math.pow(x[0] + x[1] * x[1] - 7.0, 2.0)
#     return function_value
# 
# himmelblau_x = np.linspace(-5, 5, 100)
# himmelblau_y = np.linspace(-5, 5, 100)
# himmelblau_f = np.zeros((100, 100))
# index_x = 0
# for i in himmelblau_x:
#     index_y = 0
#     for j in himmelblau_y:
#         himmelblau_f[index_y, index_x] = himmelblaufunct([i, j])
#         index_y += 1
#     index_x += 1
# 
# plt.scatter(pop.champion_x[0], pop.champion_x[1], marker='o', color='red', s=250, label='Champion')
# plt.contour(himmelblau_x, himmelblau_y, himmelblau_f, 100)
# plt.legend(loc='best')
# plt.colorbar()
# plt.xlabel('x coordinate')
# plt.ylabel('y coordinate')
# plt.xlim([-5, 5])
# plt.ylim([-5, 5])
# plt.title('Optimization of Himmelblau function')
# plt.tight_layout()
# plt.savefig(fname='./Images/himmelblau_optimization.png', bbox_inches='tight')
# plt.show()
# 
# my_dict = {'a': [5,2], 'b': [10,2],'c': [6,2] ,'d': [12,2] ,'e': [7,2]}
# print(my_dict.values())
# print(min(my_dict['d']))
# 
