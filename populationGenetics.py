import pandas as pd
import random



def initialize_population(vcf_file):
    vcf_data = pd.read_csv(vcf_file, comment='#', sep='\t', header=None)
    population = []
    for individual in vcf_data.columns[9:]:
        maternal_chrom = []
        paternal_chrom = []
        for index, row in vcf_data.iterrows():
            genotype = row[individual].split('|')
            maternal_chrom.append(int(genotype[0]))
            paternal_chrom.append(int(genotype[1]))
        population.append((maternal_chrom, paternal_chrom))
    return population


def reproduce(parent1, parent2, snps_count):
    crossover_point_m = random.randint(0, snps_count - 1)
    crossover_point_p = random.randint(0, snps_count - 1)

    # Randomly choose one chromosome from each parent
    maternal_chromosome = random.choice(parent1)
    paternal_chromosome = random.choice(parent2)

    child_maternal = maternal_chromosome[:crossover_point_m] + maternal_chromosome[crossover_point_m:]
    child_paternal = paternal_chromosome[:crossover_point_p] + paternal_chromosome[crossover_point_p:]

    return (child_maternal, child_paternal)


def fitness_function(individual, beneficial_snp_position):
    """
    Calculate the fitness based on genotype at a specific SNP position.
    Fitness is higher for heterozygous or homozygous for the beneficial allele.
    """
    maternal_allele = individual[0][beneficial_snp_position]
    paternal_allele = individual[1][beneficial_snp_position]

    if maternal_allele != paternal_allele:
        # Heterozygous at the beneficial SNP
        return 1.5
    elif maternal_allele == 1 and paternal_allele == 1:
        # Homozygous for the beneficial allele
        return 2
    else:
        # Homozygous for the non-beneficial allele
        return 1


def select_parent(population, fitness_function, beneficial_snp_position, exclude=None):
    fitness_scores = [fitness_function(individual, beneficial_snp_position) for individual in population]
    if exclude is not None:
        fitness_scores[exclude] = 0
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        return random.choice(population)
    probabilities = [score / total_fitness for score in fitness_scores]
    return random.choices(population, weights=probabilities, k=1)[0]

# Initialize the population
#population = initialize_population('initial_population.vcf')
#snps_count = len(population[0][0])

# Simulation parameters
#num_generations = 10
#population_size = len(population)

# Simulate evolution
#for generation in range(num_generations):
#     new_population = []
#     for i in range(population_size):
#         parent1 = select_parent(population, fitness_function, beneficial_snp_position)
#         parent2 = select_parent(population, fitness_function, beneficial_snp_position, exclude=population.index(parent1))
#         offspring = reproduce(parent1, parent2, snps_count)
#         new_population.append(offspring)
#     population = new_population

# The population variable now contains the evolved population
def format_population(population):
    formatted_data = []
    for individual in population:
        maternal_chrom = ''.join(map(str, individual[0]))
        paternal_chrom = ''.join(map(str, individual[1]))
        formatted_data.append(f"Maternal chrom: {maternal_chrom}\nPaternal chrom: {paternal_chrom}\n")
    return formatted_data

# Format the final population for output
#formatted_population = format_population(population)

# Output file path
#output_file_path = 'evolved_population1.txt'

# Write the formatted population data to a text file
#with open(output_file_path, 'w') as file:
#    file.writelines(formatted_population)

#output_file_path

#PART C ###########################################################################################

# Initialize the population
# population = initialize_population('initial_population.vcf')
# snps_count = len(population[0][0])  # Assuming number of SNPs is the length of the chromosome

def count_extinct_snps(population, snps_count):
    extinct_count = 0
    for snp_index in range(snps_count):
        alleles = set()
        for individual in population:
            alleles.add(individual[0][snp_index])
            alleles.add(individual[1][snp_index])
        if len(alleles) == 1:
            extinct_count += 1
    return extinct_count


def fitness_function_neutral(individual):
    return 1


def select_parent(population, fitness_function, beneficial_snp_position=None, exclude=None):
    if beneficial_snp_position is None:
        fitness_scores = [fitness_function(individual) for individual in population]
    else:
        fitness_scores = [fitness_function(individual, beneficial_snp_position) for individual in population]

    if exclude is not None:
        fitness_scores[exclude] = 0
    total_fitness = sum(fitness_scores)
    if total_fitness == 0:
        return random.choice(population)
    probabilities = [score / total_fitness for score in fitness_scores]
    return random.choices(population, weights=probabilities, k=1)[0]


# Now you can call select_parent without specifying beneficial_snp_position for neutral evolution
# parent1 = select_parent(population, fitness_function_neutral)
# parent2 = select_parent(population, fitness_function_neutral, exclude=population.index(parent1))


# Evolution with neutral fitness
#new_population = []
#for i in range(len(population)):
#    parent1 = select_parent(population, fitness_function_neutral)
#    parent2 = select_parent(population, fitness_function_neutral, exclude=population.index(parent1))
#    offspring = reproduce(parent1, parent2, snps_count)
#    new_population.append(offspring)
#population = new_population

# Count extinct SNPs
#extinct_snps = count_extinct_snps(population, snps_count)
#extinction_probability = extinct_snps / snps_count

#print("Probability of allele extinction:", extinction_probability)

'''Part D
import matplotlib.pyplot as plt
def neutral_fitness_function(individual):
    return 1

def simulate_and_track_allele_frequency(population, snps_count, num_generations=20, track_snps=100):
    allele_frequencies = {i: [] for i in range(track_snps)}

    # Record initial frequencies
    for i in range(track_snps):
        alternate_allele_count = sum(individual[0][i] + individual[1][i] for individual in population)
        frequency = alternate_allele_count / (2 * len(population))
        allele_frequencies[i].append(frequency)

    for gen in range(num_generations):
        new_population = []
        for i in range(len(population)):
            parent1 = select_parent(population, neutral_fitness_function)
            parent2 = select_parent(population, neutral_fitness_function, exclude=population.index(parent1))
            offspring = reproduce(parent1, parent2, snps_count)
            new_population.append(offspring)

        population = new_population

        for i in range(track_snps):
            alternate_allele_count = sum(individual[0][i] + individual[1][i] for individual in population)
            frequency = alternate_allele_count / (2 * len(population))
            allele_frequencies[i].append(frequency)

    return allele_frequencies


# Initialize and run the simulation
population = initialize_population('initial_population.vcf')
snps_count = len(population[0][0])
beneficial_snp_position=42
allele_frequencies = simulate_and_track_allele_frequency(population, snps_count)

# Plotting
plt.figure(figsize=(15, 8))
for i in range(100):
    plt.plot(range(0, 21), allele_frequencies[i], label=f'SNP {i+1}')  # range starts from 0 for initial generation

plt.xlabel('Generation')
plt.ylabel('Alternate Allele Frequency')
plt.title('Allele Frequency over Generations')
plt.legend(loc='upper right')
plt.show()
'''
# Part E

import matplotlib.pyplot as plt

def neutral_fitness_function(individual):
    return 1

def simulate_and_track_extinction_fixation(population, snps_count, num_generations=1000):
    extinction_probabilities = []
    fixation_probabilities = []

    for gen in range(num_generations):
        new_population = []
        for i in range(len(population)):
            parent1 = select_parent(population, neutral_fitness_function)
            parent2 = select_parent(population, neutral_fitness_function, exclude=population.index(parent1))
            offspring = reproduce(parent1, parent2, snps_count)
            new_population.append(offspring)
        population = new_population

        extinct_count = 0
        fixed_count = 0

        # Check each SNP for extinction and fixation
        for i in range(snps_count):
            alleles = [individual[chrom][i] for individual in population for chrom in range(2)]
            if all(allele == alleles[0] for allele in alleles):
                if alleles[0] == 0:  # Extinct if the fixed allele is the reference allele
                    extinct_count += 1
                else:  # Fixed if all have the same non-reference allele
                    fixed_count += 1

        extinction_prob = extinct_count / snps_count
        fixation_prob = fixed_count / snps_count

        extinction_probabilities.append(extinction_prob)
        fixation_probabilities.append(fixation_prob)

    return extinction_probabilities, fixation_probabilities

# Plotting code...
num_generations = 1000
population = initialize_population('initial_population.vcf')
snps_count = len(population[0][0])
extinction_probs, fixation_probs = simulate_and_track_extinction_fixation(population, snps_count, num_generations)
plt.figure(figsize=(15, 8))
plt.plot(range(1, num_generations + 1), extinction_probs, label='Extinction Probability')
plt.plot(range(1, num_generations + 1), fixation_probs, label='Fixation Probability')
plt.xlabel('Generation')
plt.ylabel('Probability')
plt.title('Probability of Allele Extinction and Fixation over 1000 Generations')
plt.legend()
plt.show()


