import pygmo as pg
import math

class my_minlp:

    def fitness(self, x):
        obj = 0
        for i in range(3):
            obj += (x[2*i-2]-3)**2 / 1000. - (x[2*i-2]-x[2*i-1]) + math.exp(20.*(x[2*i - 2]-x[2*i-1]))

        ce1 = 4*(x[0]-x[1])**2+x[1]-x[2]**2+x[2]-x[3]**2
        ce2 = 8*x[1]*(x[1]**2-x[0])-2*(1-x[1])+4*(x[1]-x[2])**2+x[0]**2+x[2]-x[3]**2+x[3]-x[4]**2
        ce3 = 8*x[2]*(x[2]**2-x[1])-2*(1-x[2])+4*(x[2]-x[3])**2+x[1]**2-x[0]+x[3]-x[4]**2+x[0]**2+x[4]-x[5]**2
        ce4 = 8*x[3]*(x[3]**2-x[2])-2*(1-x[3])+4*(x[3]-x[4])**2+x[2]**2-x[1]+x[4]-x[5]**2+x[1]**2+x[5]-x[0]
        ci1 = 8*x[4]*(x[4]**2-x[3])-2*(1-x[4])+4*(x[4]-x[5])**2+x[3]**2-x[2]+x[5]+x[2]**2-x[1]
        ci2 = -(8*x[5] * (x[5]**2-x[4])-2*(1-x[5]) +x[4]**2-x[3]+x[3]**2 - x[4])

        return [obj, ce1,ce2,ce3,ce4,ci1,ci2]

    def get_bounds(self):
        #return ([-5]*4 + [0,0],[5]*4 + [0,0])
        
        # print([-5]*6,[5]*6)
        # lower_bounds = [-5]
        # lower_bounds.append([-5 for i in range(4)])
        # return (lower_bounds ,[5]*6)
        return ([-5]*6,[5]*6)

    def get_nic(self):
        return 6

    def get_nix(self):
        return 2

    def gradient(self, x):
        return pg.estimate_gradient_h(lambda x: self.fitness(x), x)

prob = pg.problem(my_minlp())
#prob.c_tol = [1e-14]*4 + [0]*2
prob.c_tol = [1e-8]*6
print(prob)

#algo = pg.algorithm(pg.ihs())
#pop = pg.population(prob, size=10000)
## x_best = pop.get_x()
## x_best[-1] = 0; x_best[-2] = 0
## pop.push_back(x_best)
#
#fitness_list=  []
#population_list=  []
#for i in range(5000):
#    pop = algo.evolve(pop)
#    fitness_list.append(pop.get_f())
#    population_list.append(pop.get_x())
#    print(pop.champion_f)
#    print('Evolving population; at generation ' + str(i))
