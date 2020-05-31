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
            print(population)
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
    starting_population = [np.array([[ 0.35124206,  0.18364787,  0.1922011 , -2.21746278,  0.5715741 ,
         0.17514459, -0.06177887, -1.06592961, -0.36305876, -0.97099295,
        -0.47544896,  1.76198191,  0.30096455, -0.32830829,  1.94622403],
       [-0.54891099, -0.23203806,  0.83454824, -2.24766875,  0.93882542,
        -0.62871016,  1.27529598,  0.36552926,  0.58169327,  1.32130783,
        -1.52574307,  0.4501985 ,  0.08082787, -1.04572819,  0.38057286]]), np.array([[ 1.95036502, -0.898277  , -0.01523718, -0.96874174,  0.48040115,
         0.19486922,  0.30420499,  1.0024501 , -0.80335476, -1.04580682,
         1.7920282 , -1.12402349, -0.72981733, -0.88441944,  0.74535907],
       [ 1.78222402,  1.08397593,  0.74764167, -0.85760906, -0.47381623,
        -1.03196518,  0.62396387,  1.5108658 , -0.78925764,  0.44245452,
        -0.33540585, -2.00927206,  0.04643864, -0.09789315, -0.64234528]]), np.array([[ 0.4563265 , -1.23997047, -0.96336271, -1.14383322,  1.81648077,
         1.09883244, -0.11108897, -1.31052194,  0.52841341,  0.62494439,
         0.56338792,  1.50322291, -0.61450101, -0.14239122, -1.06593881],
       [-0.12773556,  0.64947238,  0.55637913, -1.20311895,  1.0112516 ,
        -1.171394  ,  0.89342546, -1.13533939,  0.29128778,  1.50199429,
         0.40941052,  0.91722684, -2.18904062, -0.42509862,  0.02127914]]), np.array([[ 0.97033125,  0.65251806,  1.3184504 , -0.96946098,  0.84965086,
        -1.46658839,  1.12835623, -0.9056667 ,  1.42042677,  0.60598973,
        -1.07956374, -0.53002505,  0.07270873, -1.12099703, -0.94613017],
       [ 1.57637955,  1.4798016 , -0.1627499 ,  0.34839497,  0.29227293,
        -0.42008845, -1.07014557, -0.72321676,  0.76757337, -0.02573341,
        -0.38996991, -1.5545025 ,  0.96738702, -1.87760283,  0.79219988]]), np.array([[-0.12489625,  0.34939806, -0.70346054,  1.40534449, -0.25035919,
        -0.03218514,  1.54954959, -0.27304233,  0.24730126,  1.14049754,
        -0.48325423,  0.62587777, -0.95071764,  0.13506404, -2.63511744],
       [ 0.61836687, -0.55111188, -1.52566442,  1.10353729, -0.87925347,
        -1.18415365, -0.37103602,  1.01160104,  0.41193547,  1.56600339,
        -0.16340158, -0.84257491,  0.89899567,  1.23027086, -1.32351466]]), np.array([[-0.48484472, -0.58965507,  0.75579641,  0.92401302,  1.17550053,
         0.0437335 , -1.75422164,  0.72243786,  0.62163489,  1.21285285,
        -0.55984178,  0.24014731, -0.4658091 ,  0.46976737, -2.31151143],
       [-0.12988812, -0.00994683, -1.25322819,  2.24511708,  1.59317309,
         0.63693512, -0.99131816, -0.4341244 ,  1.31214508, -0.85127371,
        -0.40810914, -0.19253871, -1.20153477,  0.08040617, -0.39581451]]), np.array([[-0.32612331,  1.10028729,  0.34514793, -0.71088714,  1.83232358,
        -1.55078603, -0.71917256, -0.79770971, -0.62059635,  1.08664011,
         0.29769579, -0.57555051,  1.7287752 , -0.00927165, -1.08077264],
       [ 0.73947435, -0.08606624,  1.31181405, -0.14974518, -0.35024211,
        -0.40805488,  0.73996095, -1.16284249,  0.21188261,  1.43659252,
         0.73733061, -0.90256255,  0.22131785,  0.29044798, -2.62930748]]), np.array([[-1.02767356, -0.68557497,  1.00403742, -1.51679477,  0.70657311,
        -1.29361751,  1.27247207,  1.26716192,  0.49399766, -1.45900436,
         0.09818942,  0.21752209, -0.78171614,  0.48191736,  1.22251026],
       [-1.09725406, -0.13018149,  0.74746757, -1.26693173,  1.3242449 ,
         1.61735756, -0.77173704, -0.0203396 ,  0.70763861, -1.47067865,
         0.18673899, -1.30382289, -0.51223886,  1.06346979,  0.92626691]]), np.array([[-1.60884469,  0.66926224, -0.01536788, -0.86145521,  1.28230615,
         1.63192291,  0.19740201, -1.5358598 , -0.73829293, -0.38074462,
        -0.40152965, -0.96251094,  0.65551154,  0.74261099,  1.32558986],
       [-1.20977913,  0.54622818,  0.15215429, -0.9808931 ,  0.38405083,
        -0.47553349,  1.14576729,  1.06546868, -0.56723313,  0.60443653,
        -0.77202941, -1.59242005,  0.34830828, -0.83714849,  2.18862273]]), np.array([[-1.00154476, -0.39987715,  0.09504029, -0.0700659 ,  1.0602595 ,
         1.30050091, -1.09352077, -1.84448557,  1.2408776 ,  0.90926318,
         1.07419508,  0.09693815,  0.76900164, -1.0134604 , -1.12312183],
       [-1.83611406,  0.23181558,  0.55125202,  1.15295463,  1.01299528,
        -0.48665252, -1.12130271, -0.27099123,  0.02636121,  0.37919163,
         1.20600989,  1.24350851, -0.97119371, -1.7066909 ,  0.58885637]]), np.array([[ 1.33890798, -0.16845233, -0.68130736,  0.48385616, -1.19246304,
         0.0729942 , -0.21728944,  0.00720998, -1.09677327,  1.76363891,
        -1.36263118,  0.05199583,  1.84332777,  0.3133721 , -1.15638633],
       [ 0.62257558,  0.29011485, -1.54202282,  0.4657507 ,  0.16965782,
         1.17361121, -0.789318  , -1.03167741,  0.98666141, -0.64984507,
        -1.24092512,  1.33974737,  1.48542609,  0.0963112 , -1.37606781]]), np.array([[ 0.69195564, -0.99690295,  0.45731401,  1.17536369, -1.11244555,
        -1.06351976,  1.13727875,  0.10854819, -0.64316329, -1.08547647,
        -0.43831984,  0.93302186, -0.5042616 , -0.82226208,  2.1628694 ],
       [-0.2062092 , -1.18999241, -0.86948269,  1.15265517, -0.80387329,
         0.5399064 ,  2.18791485,  1.38352856, -0.4590354 ,  0.91967788,
        -0.49388615, -0.46716332, -0.55587468, -1.38839799,  0.25023228]]), np.array([[ 0.63824784,  1.32253871, -0.33867343,  1.23228283,  0.62017802,
         0.39729612, -0.48272777,  0.07287643, -1.49158123, -1.36266706,
         0.32461674,  1.39609659,  0.51030655, -1.37701353, -1.46177683],
       [ 0.4765043 ,  2.07042661,  0.6391383 , -0.89636996,  1.36289717,
        -0.19617197,  0.0800701 ,  0.95606704, -1.32233374, -1.12461019,
        -1.11020499, -0.92026974,  0.84519162, -0.8047671 , -0.05556745]]), np.array([[ 1.30967114, -1.28006553, -1.61310586,  1.63485164, -0.46708038,
        -0.058168  , -1.05345789, -1.33283121,  0.53707038,  0.71878989,
         0.73086927,  0.60414596,  0.21396661,  0.8971558 , -0.84181184],
       [-0.6296094 ,  0.11669337,  0.11811203,  1.44216507, -1.33140867,
         1.17007531,  0.24632516, -1.47607738,  0.21020693,  0.41993883,
        -0.84974822, -0.45747815,  0.58364286,  1.88094241, -1.44378015]]), np.array([[-0.92214144,  0.60533353, -0.18105195, -1.55442283, -0.1565325 ,
         1.6413328 , -1.0315531 , -0.01505336,  1.45467486, -0.81811235,
         1.09824202, -0.47633692,  0.0402108 ,  1.43864906, -1.12323861],
       [ 0.50787287,  0.54588777,  0.58402602, -0.0060654 ,  1.26355403,
         2.22776298, -1.0098926 ,  0.22470083, -0.7836571 , -0.68335597,
         0.86428105, -0.62740749, -1.58800546, -0.22372819, -1.29597335]]), np.array([[-0.27868079,  0.58257503, -0.04030903, -1.18800688, -0.35743634,
        -0.72062623,  1.41665594,  1.49265139, -1.68471802, -0.04068829,
         1.64350783, -1.31713628, -0.57204355,  0.28869039,  0.77556482],
       [-0.52896429,  0.07060053,  0.9894927 , -0.94210031, -0.74738211,
        -1.18385192, -0.4654378 , -0.45364657, -1.80266008,  0.86495525,
         0.98931224,  0.04399549,  0.74183077,  2.21187636,  0.21197973]]), np.array([[ 0.93414297, -1.43814199, -0.21840885,  0.86260739,  0.32096151,
        -1.61136885,  0.38068923,  0.38329493, -1.27902094,  0.35520791,
        -1.90170366,  1.06817509,  0.53100447,  0.51108427,  1.10147652],
       [ 0.39700821, -1.11678967,  1.99012914, -0.79380126, -0.74750939,
        -1.4782589 ,  0.15742041, -0.7629436 ,  0.35112351,  1.49529551,
        -0.70084544,  0.98034872, -0.72783774, -0.17540339,  1.13206388]]), np.array([[ 0.50794392,  0.14921867, -0.85619737,  0.65519327, -1.1071847 ,
         0.97510548,  0.9474698 , -1.49453462,  1.12020627,  0.29539686,
        -1.2105347 , -1.10098673, -1.13648388,  0.6834623 ,  1.57192541],
       [-0.80620421,  1.52271546, -1.04665823,  1.32588956,  0.67272928,
        -0.47252315, -0.06404297, -0.94369664, -0.497424  , -0.49910978,
         0.51229805, -1.55976892, -0.84749951,  1.49889069,  1.20440435]]), np.array([[-0.9192588 ,  0.64561038, -0.57620294,  0.36221688, -0.67906313,
         1.18918627,  1.0058768 ,  1.34457311,  1.38419033, -0.79385301,
        -1.23512561, -0.33081371,  0.91274793, -1.82491965, -0.48516486],
       [-1.0456506 ,  0.05841701, -0.64719869, -0.25150827,  1.08482044,
         1.52710528, -0.43054231,  0.28496758,  1.05549135,  0.48249836,
        -0.00522083, -2.33357361,  0.24405251, -1.19363269,  1.16997446]]), np.array([[ 0.94907807,  1.25754085, -0.01920401, -0.46747145,  1.28626474,
        -0.76925427,  0.74159817, -1.4871829 ,  0.02787697, -0.70359098,
         1.16419049,  0.00534946,  0.99179373, -1.37677633, -1.60021255],
       [ 1.08017515,  1.16448744, -0.17528641, -1.5987907 ,  1.27104539,
        -0.57958995,  0.18527344,  0.34858624, -1.07140826,  0.3629247 ,
         1.22375503,  0.83518338, -1.83266629, -0.93325688, -0.28043227]])]

    main(args.iterations, args.population, args.pcrossover, args.pmutation, args.mutationsd,
        REFERENCE, verbose=args.verbose, plot=True, starting_population=starting_population)

if __name__ == '__main__':
    run()