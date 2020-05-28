'''
This program runs the genetic algorithm in same-stats with a number of different parameters for optimization purposes.
Parameters to tune include:
    * number of iterations
    * population size
    * probability of crossover
    * probability of mutation
    * mutation standard deviation (magnitude)
'''

import dissimilarity as ds
import same_stats as ss
from itertools import product
from time import perf_counter
from functools import partial

'''test_iterations = (10, 50, 150)
test_population_sizes = (20, 50, 100, 200, 500)
test_p_crossover = (0.5, 0.7, 0.9, 1)
test_p_mutation = (0, 0.01, 0.05)
mutation_sd = 0.25'''
# Not testing changes in mutation standard deviation, since mutations are rare occurrences anyway

test_iterations = (10,50)
test_population_sizes = (20,100,200)
test_p_crossover = (0.5,0.7,0.9,1)
test_p_mutation = (0,0.01,0.05)
mutation_sd = 0.25

performance = {}
measures_to_test = (ds.data_diff, ds.skewness_diff, ds.kurt_diff, ds.power_diff)
REFERENCE = ss.load_data('dataset1.txt')

for combination in product(test_iterations, test_population_sizes, test_p_crossover, test_p_mutation):
    start_time = perf_counter()
    populations = ss.run_genetic(*combination, mutation_sd,
        measures_to_test, REFERENCE)
    best_distributions = [pop[0] for pop in populations[1:]]
    max_fitnesses = [partial(pair[0], REFERENCE)(pair[1]) for pair in zip(measures_to_test, best_distributions)]
    end_time = perf_counter()

    performance[combination] = (max_fitnesses, end_time - start_time)

print(performance)