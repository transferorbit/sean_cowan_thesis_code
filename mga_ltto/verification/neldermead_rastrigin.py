# if __name__ == '__main__': #to prevent this code from running if this file is not the source file.
import pygmo as pg
import multiprocessing as mp

seed = 421
pop_size = 100000

problem = pg.rastrigin(10)
print(problem.best_known())
mp.freeze_support()
my_algorithm = pg.algorithm(pg.nlopt(solver='neldermead'))
pop = pg.population(problem, size=pop_size, seed=seed)
for i in range(100):
    pop = my_algorithm.evolve(pop)
fitness = pop.get_f()[pop.best_idx()]
print(pop.best_idx())
print(fitness)

# archi = pg.archipelago(n=1, algo = my_algorithm, prob=problem, pop_size = pop_size, seed=seed)#, udi = my_island)
#
# list_of_f_dicts = []
# list_of_x_dicts = []
# print('Evolving ..')
# archi.evolve()
# champions_x = archi.get_champions_x()
# champions_f = archi.get_champions_f()
# archi.wait_check()
# print('Evolution finished')
# print(champions_x, champions_f)
