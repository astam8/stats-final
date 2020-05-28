# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, ks_2samp
from math import sqrt
from functools import partial
import dissimilarity as ds
import argparse

# Default number of iterations to run the genetic algorithm, if no argument is specified
DEFAULT_ITERATIONS = 150
DEFAULT_POPSIZE = 350

parser = argparse.ArgumentParser()
parser.add_argument('filename', help='specify the filename to read data from')
parser.add_argument('-i', '--iterations', type=int, default=DEFAULT_ITERATIONS,
    help='the number of iterations to run each genetic algorithm')
parser.add_argument('-p', '--population', type=int, default=DEFAULT_POPSIZE,
    help='size of population of genes for genetic algorithm')


args = parser.parse_args()
print(args.filename, args.iterations, args.population)

def load_data(filename):
    with open(filename, 'r') as fin:
        lines = fin.read().split('\n')
        x, y = zip(*[line.split(' ') for line in lines])
        x = [float(val) for val in x]
        y = [float(val) for val in y]
        
    return np.array([x, y])

REFERENCE = load_data(args.filename)

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

REF_XBAR, REF_YBAR, REF_SX, REF_SY, REF_R, REF_N = summary(REFERENCE)

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

# L is the matrix ((1, 0), (r, sqrt(1 - r^2))). When multiplied by an uncorrelated 2 x n matrix, the correlation of the two
# rows becomes r.

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

    # Obtain the correct correlation matrix
    L = np.array([
        [1, 0],
        [r, sqrt(1 - r ** 2)]
    ])
    
    # Multiply correlation matrix by A to force correct value of r
    A = np.matmul(L, A)
    
    # Set mean and variances of data to reference distribution
    A = np.array([
        fix_stats(A[0], xbar, sx),
        fix_stats(A[1], ybar, sy)
    ])
    
    return A

# Following method generates a set of REF_N datapoints with almost identical summary statistics to the reference
def replicate_statistics(ref):
    xbar, ybar, sx, sy, r, n = summary(ref)
    # Generate random normal data
    return fit_summary_statistics_of_to(rand_normal_data(n), ref)

def crossover(data1, data2, p_crossover):
    if np.random.rand() < p_crossover:
        crossover_point = np.random.randint(data1.shape[1] + 1) # pick random column for crossover
        left1 = data1[:, :crossover_point]
        right1 = data2[:, crossover_point:]

        right2 = data1[:, crossover_point:]
        left2 = data2[:, :crossover_point]
        return (np.concatenate((left1, right1), axis=1), np.concatenate((left2, right2), axis=1))
    else:
        return (data1, data2)

def mutate(data, p_mutation):
    if np.random.rand() < p_mutation:
        MUTATE_SD = 0.25
        return data + np.random.normal(0, MUTATE_SD, data.shape)
    else:
        return data

def genetic_step(population, fitness, p_crossover, p_mutation):
    # Calculate fitnesses
    fitnesses = [fitness(gene) for gene in population]
    print(max(fitnesses))
    sum_fitness = sum(fitnesses)
    fitnesses_distribution = [x / sum_fitness for x in fitnesses]

    # Create new population first
    new_population = []

    # Elitism
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
        offspring_1, offspring_2 = crossover(*parents, p_crossover)
        new_population.append(offspring_1)
        new_population.append(offspring_2)

    # Mutation
    new_population = [mutate(gene, p_mutation) for gene in new_population]

    # We may need to trim randomly from new_population to keep population sizes stable.
    if len(new_population) > len(population):
        # We prefer to cut off offspring
        offset = len(population) - len(new_population)
        new_population = new_population[:offset]

    return new_population

# Plotting
fig = plt.figure()

all_populations = [[REFERENCE]]

population = [replicate_statistics(REFERENCE) for i in range(args.population)]
population = [fit_summary_statistics_of_to(gene, REFERENCE) for gene in population]

measures_to_test = (ds.data_diff, ds.skewness_diff, ds.kurt_diff, ds.power_diff)

for diff in measures_to_test:
    population = [replicate_statistics(REFERENCE) for i in range(args.population)]
    print('––––STARTING GENETIC FOR DISSIMILARITY MEASURE:', diff)
    for i in range(args.iterations):
        print(i)
        population = genetic_step(population, partial(diff, REFERENCE), 0.8, 0.01)
        # Stat fix
        population = [fit_summary_statistics_of_to(gene, REFERENCE) for gene in population]
    all_populations.append(population)
    print('population:', summary(population[0]))
    print(population[0])
    print('reference:', summary(REFERENCE))

for i in range(len(measures_to_test) + 1):
    ax = plt.subplot(2, 3, i + 1)
    ax.scatter(all_populations[i][0][0], all_populations[i][0][1], s=5)

plt.ioff()
plt.show()