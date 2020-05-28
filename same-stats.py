# Imports
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, ks_2samp
from math import sqrt
from functools import partial
import dissimilarity as ds

REFERENCE = np.array([[-0.83375003,  1.0954705 , -0.05082271, -0.22376648,  0.36000685,
        -0.09391634,  0.2156405 , -0.93300305, -1.54057091, -0.48787898,
        -1.67570942,  0.17950772,  0.35156441,  1.60994239,  0.025276  ,
         2.57038455,  0.55633747, -0.86154288, -0.88514176,  0.62197216],
       [-1.28014033,  0.50484594,  0.42892175, -1.14352745,  0.78320453,
         0.40341267,  0.88552499, -1.26098764, -0.48010823, -1.08377086,
        -0.49397959,  0.21351545, -0.29116993,  1.72080078, -0.0695087 ,
         1.05075295,  1.08878165, -1.1260957 , -0.13119122,  0.28071896]])

'''REFERENCE = np.array([[-0.26726124, -0.26726124, -0.26726124, -0.26726124, -0.26726124,
        -0.26726124, -0.26726124, -0.26726124, -0.26726124, -0.26726124,
        -0.26726124, -0.26726124, -0.26726124, -0.26726124,  3.74165739],
       [-1.33808113, -1.11256184, -0.88704255, -0.66152326, -0.43600397,
        -0.21048467,  0.01503462,  0.24055391,  0.4660732 ,  0.6915925 ,
         0.91711179,  1.14263108,  1.36815037,  1.59366967, -1.78911972]])'''

'''REFERENCE = np.array([[ 1.43375611,  2.04441281,  1.41975531, -0.0815076 , -0.36698188,
        -0.36245928,  0.11546782, -0.7661847 , -0.88579085, -1.94706537,
         0.575897  , -0.29487172,  0.00660205, -0.93777525,  0.04674554],
       [-0.031157  ,  0.75224918,  1.28054373, -1.43668944,  1.12769705,
        -0.41057733,  0.32079744, -1.31533635, -1.76772165, -0.72650318,
         0.53151029,  0.90611511,  0.45094862,  1.26491801, -0.94679448]])'''
# Number of datasets to randomly generate
NUM_DATASETS = 10

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

# Generate random data from a normal distribution
def rand_normal_data(n):
    A = np.random.rand(2, n)
    A[0] = fix_stats(A[0], 0, 1)
    A[1] = fix_stats(A[1], 0, 1)
    return A

# L is the matrix ((1, 0), (r, sqrt(1 - r^2))). When multiplied by an uncorrelated 2 x n matrix, the correlation of the two
# rows becomes r.
L = np.array([
    [1, 0],
    [REF_R, sqrt(1 - REF_R ** 2)]
])

# Following method generates a set of REF_N datapoints with almost identical summary statistics to the reference
def replicate_statistics(ref):
    xbar, ybar, sx, sy, r, n = summary(ref)
    # Generate random normal data
    A = rand_normal_data(n)
    
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

# Genetic approach

def fit_summary_statistics_of_to(data, ref):
    xbar, ybar, sx, sy, r, n = summary(ref)
    # Generate random normal data
    A = fix_stats(data, 0, 1)
    
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


ITERATIONS = 150
POP_SIZE = 350

# Plotting
fig = plt.figure()

all_populations = [REFERENCE]

population = [replicate_statistics(REFERENCE) for i in range(POP_SIZE)]
population = [fit_summary_statistics_of_to(gene, REFERENCE) for gene in population]

for diff in (ds.data_diff, ds.skewness_diff, ds.kurt_diff, ds.power_diff):
    population = [replicate_statistics(REFERENCE) for i in range(POP_SIZE)]
    print('––––STARTING GENETIC FOR DISSIMILARITY MEASURE:', diff)
    for i in range(ITERATIONS):
        print(i)
        population = genetic_step(population, partial(diff, REFERENCE), 0.8, 0.01)
        # Stat fix
        population = [fit_summary_statistics_of_to(gene, REFERENCE) for gene in population]
    all_populations.append(population)
    print('population:', summary(population[0]))
    print(population[0])
    print('reference:', summary(REFERENCE))

for i in range(5):
    ax = plt.subplot(2, 3, i + 1)
    ax.scatter(all_populations[i][0], all_populations[i][1], s=5)
#plt.scatter(REFERENCE[0], REFERENCE[1])
#plt.scatter(population[0][0], population[0][1])



plt.ioff()
plt.show()