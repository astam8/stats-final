# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, ks_2samp
from math import sqrt
from functools import partial
import dissimilarity as ds
import argparse

# Default number of iterations to run the genetic algorithm, if no argument is specified
DEFAULT_ITERATIONS = 125
# Default population size for the genetic algorithm, if no argument is specified
DEFAULT_POPSIZE = 300
# Default probability of crossover for the genetic algorithm, if no argument is specified
DEFAULT_PCROSSOVER = 0.7
# Default probability of mutation for the genetic algorithm, if no argument is specified
DEFAULT_PMUTATION = 0.01
# Default standard deviation for mutations for the genetic algorithm; increasing will increase the likelihood of large mutations
DEFAULT_MUTATIONSD = 0.25

def load_data(filename):
    with open(filename, 'r') as fin:
        lines = fin.read().split('\n')
        x, y = zip(*[line.split(' ') for line in lines])
        x = [float(val) for val in x]
        y = [float(val) for val in y]
        
    return np.array([x, y])

# Get summary statistics of a list of data
def summary(data):
    x, y = data
    xbar = np.mean(x)
    ybar = np.mean(y)
    sx = np.std(x)
    sy = np.std(y)
    r = pearsonr(x, y)[0]
    n = data.shape[1]
    return (xbar, ybar, sx, sy, r, n)

# Fix mean and standard deviation of a list of data
def fix_stats(data, mean, std):
    data = (data - np.mean(data)) / np.std(data)
    data = (data * std) + mean
    return data

# Generate random 2-variable data with n data points with mean 0 and standard deviation 1, and return as an (2 x n) np.array
def rand_normal_data(n):
    A = np.random.rand(2, n)
    for row in A:
       row[:] = fix_stats(row, 0, 1)
    return A

# Fit the summary statstics of "data" to match those of "ref"
def fit_summary_statistics_of_to(data, ref):
    xbar, ybar, sx, sy, r, n = summary(ref)

    A = data
    
    # Orthogonalize A
    A[1] = A[1] - (np.dot(A[0], A[1]) / (np.linalg.norm(A[0]) ** 2))*A[0]
    
    # Normalize A
    A = np.array([
        fix_stats(A[0], 0, 1),
        fix_stats(A[1], 0, 1)
    ])

    # Obtain the correct correlation matrix to multiply by
    L = np.array([
        [1, 0],
        [r, sqrt(1 - r ** 2)]
    ])
    
    # Multiply correlation matrix by A to force correct value of r
    A = L @ A
    
    # Set mean and variances of data to reference distribution
    A = np.array([
        fix_stats(A[0], xbar, sx),
        fix_stats(A[1], ybar, sy)
    ])
    
    return A

# Following method generates a set of REF_N datapoints with almost identical summary statistics to the reference
def replicate_statistics(ref):
    xbar, ybar, sx, sy, r, n = summary(ref)
    # Generate random normal data, then fit summary statistics
    return fit_summary_statistics_of_to(rand_normal_data(n), ref)

def crossover(data1, data2, p_crossover):
    if np.random.rand() < p_crossover:
        crossover_point = np.random.randint(data1.shape[1] + 1) # pick random column for crossover
        first_left_slice = data1[:, :crossover_point]
        second_right_slice = data2[:, crossover_point:]
        first_offspring = np.concatenate((first_left_slice, second_right_slice), axis=1)

        first_right_slice = data1[:, crossover_point:]
        second_left_slice = data2[:, :crossover_point]
        second_offspring = np.concatenate((first_right_slice, second_left_slice), axis=1)
        return (first_offspring, second_offspring)
    else:
        return (data1, data2)

def mutate(data, p_mutation, mutation_sd):
    if np.random.rand() < p_mutation:
        return data + np.random.normal(0, mutation_sd, data.shape)
    else:
        return data

def genetic_step(population, fitness, p_crossover, p_mutation, mutation_sd, verbose=False):
    # Calculate fitnesses
    fitnesses = [fitness(gene) for gene in population]
    if verbose:
        print(max(fitnesses))
    sum_fitness = sum(fitnesses)
    fitnesses_distribution = [x / sum_fitness for x in fitnesses]

    # Create new population first
    new_population = []

    # Elitism: best-performing gene gets passed down to new population automatically
    max_fitness_idx = np.argmax(fitnesses)
    new_population.append(population[max_fitness_idx])

    # Selection
    selected_gene_indices = np.random.choice(range(len(population)), size=len(population), p=fitnesses_distribution)
    selected_genes = [population[idx] for idx in selected_gene_indices]

    # Crossover
    while len(new_population) < len(population):
        # make random choice of parents from selected_genes
        parent_indices = np.random.choice(range(len(selected_genes)), size=2)
        parents = [selected_genes[idx] for idx in parent_indices]
        offspring = crossover(*parents, p_crossover)
        new_population.extend(offspring)

    # Mutation
    new_population = [mutate(gene, p_mutation, mutation_sd) for gene in new_population]

    # We may need to trim randomly from new_population to keep population sizes stable, since each crossover step
    # produces 2 offspring, but elitism places 1 offspring in the population. Without this step, population size would
    # increase by 1 after each genetic step
    if len(new_population) > len(population):
        # We prefer to cut off offspring, not the elite gene (which is guaranteed to be at index 0)
        offset = len(population) - len(new_population)
        new_population = new_population[:offset] # Cuts off number of elements equal to offset

    return new_population

def run_genetic(iterations, population_size, p_crossover, p_mutation, mutation_sd, measures_to_run, ref_distribution, verbose=False, starting_population=None):
    all_populations = [[ref_distribution]]
    
    for diff_measure in measures_to_run:
        if starting_population:
            population = starting_population
        else:
            population = [replicate_statistics(ref_distribution) for i in range(population_size)]
        if verbose:
            print('––STARTING GENETIC FOR DISSIMILARITY MEASURE:', diff_measure)

        for i in range(iterations):
            if verbose:
                print('Iteration', i)
            population = genetic_step(population, partial(diff_measure, ref_distribution), p_crossover, p_mutation, mutation_sd, verbose=verbose)
            population = [fit_summary_statistics_of_to(gene, ref_distribution) for gene in population]

        population.sort(key=partial(diff_measure, ref_distribution), reverse=True)
        all_populations.append(population)

        if verbose:
            print('reference summary:', summary(ref_distribution))
            print('leading distribution summary:', summary(population[0]))
            print('leading distribution:', population[0])

    return all_populations

def main(iterations, population_size, p_crossover, p_mutation, mutation_sd, ref_distribution,
    measures_to_test=(ds.data_diff, ds.skewness_diff, ds.kurt_diff, ds.power_diff), verbose=False, plot=False,
    starting_population=None):
    
    populations = run_genetic(iterations, population_size, p_crossover, p_mutation, mutation_sd,
        measures_to_test, ref_distribution, verbose=verbose, starting_population=starting_population)
    
    if plot:
    # Plotting
        fig = plt.figure()

        for i in range(len(populations)):
            ax = plt.subplot(2, 3, i + 1)
            ax.scatter(populations[i][0][0], populations[i][0][1], s=5)

        plt.ioff()
        plt.show()

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename', help='specify the filename to read data from')
    parser.add_argument('-i', '--iterations', type=int, default=DEFAULT_ITERATIONS,
        help='the number of iterations to run each genetic algorithm')
    parser.add_argument('-p', '--population', type=int, default=DEFAULT_POPSIZE,
        help='size of population of genes for genetic algorithm')
    parser.add_argument('-c', '--pcrossover', type=int, default=DEFAULT_PCROSSOVER,
        help='probability of crossover for genetic algorithm')
    parser.add_argument('-m', '--pmutation', type=int, default=DEFAULT_PMUTATION,
        help='probability of mutation for genetic algorithm')
    parser.add_argument('-s', '--mutationsd', type=int, default=DEFAULT_MUTATIONSD,
        help='standard deviation of normal distribution for mutations for genetic algorithm')
    parser.add_argument('-v', '--verbose', default=False,
        help='print statements while running genetic algorithm', action='store_true')

    args = parser.parse_args()

    REFERENCE = load_data(args.filename)
    
    main(args.iterations, args.population, args.pcrossover, args.pmutation, args.mutationsd,
        REFERENCE, measures_to_test=(ds.data_diff, ds.skewness_diff, ds.kurt_diff, ds.power_diff), verbose=args.verbose, plot=True)

if __name__ == '__main__':
    run()