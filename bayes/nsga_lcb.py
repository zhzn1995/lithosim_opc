import GPy
import numpy as np
import sys
import matplotlib.pyplot as plt
#from platypus import NSGAII, MOEAD, Problem, Real, SPEA2, NSGAIII, Solution, InjectedPopulation, Archive
from platypus.problems import Problem
from platypus.algorithms import NSGAII
import mdenas
import nas_graph
import genetic_algorithm as ga

## load or initinal data here
dim      = 24      # input's dim
num_init = 4
xs       = mdenas.rand_arch(num_init)
ys       = np.zeros(num_init)
for i in range(num_init):
    ys[i, :] = mdenas.predict_acc(xs[i], thread = 4)
max_eval = 100

for cnt in range(max_eval):
    y1 = ys.reshape(len(ys), 1)
    gp_m1 = GPy.models.GPRegression(mdenas.encode(xs), y, Gpy.kern.RBF(input_dim = dim, ARD = True))
    gp_m1.kern.variance = np.var(y1)
    gp_m1.kern.lengthscale = np.std(xs, 0)
    gp_m1.likelihood.variance = 1e-2 * np.var(y1)
    gp_m1.optimize()

    def lcb(x):
        py1, ps2_1 = gp_m1.predict(np.array([mdenas.encode(x)]))
        ps_1       = np.sqrt(ps2_1)
        lcb1       = py1 - 3 * ps_1
        return lcb1[0, 0]
    
    ga_result = ga.run(1000, mdenas.rand_arch(50), population = 50, fix_fun = lcb)

    idx = np.random.permutation(len(ga_result))[:4]
    new_x = ga_result[idx]
    new_y = np.zeros(4)
    for i in range(4):
        new_y[i] = mdenas.predict_acc(xs[i], thread = 4)
    
    py1, ps2_1 = gp_m1.predict(new_x.reshape(1, dim))
    print(cnt)
    print('-'*60)