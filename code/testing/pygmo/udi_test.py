import pygmo as pg
import multiprocessing as mp

if __name__ == '__main__':
    mp.freeze_support()
    my_population = pg.population(pg.rosenbrock(), size = 100000, seed=33333)
    my_algorithm = pg.algorithm(pg.gaco())

    my_island = pg.mp_island()
    my_island.init_pool()
    print(my_island.run_evolve(algo=my_algorithm, pop=my_population))
    print(my_island.get_extra_info())

archi = pg.archipelago(n=5, algo = pg.gaco(100), prob = mga_low_thrust_problem, pop_size = 20, my_udi = my_island)
