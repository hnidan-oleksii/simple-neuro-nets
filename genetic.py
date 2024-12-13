import numpy as np
import random
import sys

class GeneticAlgorithm:
    def __init__(self,
                 population_size: int,
                 variables_number: int,
                 min_value: int,
                 max_value: int,
                 generations: int,
                 cross_prob: float = 0.7,
                 mutation_prob: float = 0.01,
                 bits_per_gene: int = 10):  # Added parameter for bits per gene
        self.population_size: int = population_size
        self.variables_number: int = variables_number
        self.min_value: int = min_value
        self.max_value: int = max_value
        self.generations: int = generations
        self.cross_prob: float = cross_prob
        self.mutation_prob: float = mutation_prob
        self.bits_per_gene: int = bits_per_gene  # Number of bits per weight

    def generate_population(self, population_size: int):
        population = []
        for _ in range(population_size):
            individual = []
            for _ in range(self.variables_number):
                variable = np.random.uniform(self.min_value, self.max_value)
                chromosome_segment = self.float_to_bin(variable, self.bits_per_gene)
                individual.extend([int(bit) for bit in chromosome_segment])
            population.append(individual)
        return population

    def float_to_bin(self, num, bits: int):
        normalized_num = (num - self.min_value) / (self.max_value - self.min_value)
        max_int = 2 ** bits - 1
        scaled_value = int(normalized_num * max_int)
        return format(scaled_value, f'0{bits}b')

    def bin_to_float(self, chromosome):
        genes_per_variable = self.bits_per_gene
        variables = []

        for i in range(self.variables_number):
            segment = chromosome[i * genes_per_variable:(i + 1) * genes_per_variable]
            binary_string = ''.join(map(str, segment))
            int_value = int(binary_string, 2)
            max_int = 2 ** genes_per_variable - 1
            normalized_value = int_value / max_int
            float_value = self.min_value + normalized_value * (self.max_value - self.min_value)
            variables.append(float_value)

        return variables

    def crossover(self, parent1, parent2):
        if random.random() < self.cross_prob:
            point = random.randint(1, len(parent1) - 1)
            return (parent1[:point] + parent2[point:], parent2[:point] + parent1[point:])
        return parent1, parent2

    def mutate(self, chromosome):
        return [gene if random.random() > self.mutation_prob else 1 - gene for gene in chromosome]

    def evaluate_fitness(self, population, fitness_func):
        return [fitness_func(self.bin_to_float(individual)) for individual in population]

    def selection(self, population, fitness_values):
        inverted_fitness_values = [1 / f if f != 0 else sys.float_info.max for f in fitness_values]
        fitness_sum = sum(inverted_fitness_values)
        probabilities = [f / fitness_sum for f in inverted_fitness_values]

        selected_indices = np.random.choice(len(population), size=len(population), p=probabilities)
        return [population[i] for i in selected_indices]

    def run(self, fitness_func):
        population = self.generate_population(self.population_size)

        for generation in range(self.generations):
            fitness_values = self.evaluate_fitness(population, fitness_func)
            selected_population = self.selection(population, fitness_values)

            next_generation = []
            for i in range(0, len(selected_population), 2):
                parent1 = selected_population[i]
                parent2 = selected_population[i + 1]
                offspring1, offspring2 = self.crossover(parent1, parent2)
                next_generation.extend([self.mutate(offspring1), self.mutate(offspring2)])

            population = next_generation

        best_individual = min(population, key=lambda ind: fitness_func(self.bin_to_float(ind)))
        return best_individual

