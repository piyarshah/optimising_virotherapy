# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


time_points = np.array([0, 1, 4, 8, 12, 15, 19, 22, 26, 43, 50, 59])
time_points_in_hours = time_points * 24

tumor_volumes = np.array([2.47, 2.61, 2.62, 4.82, 2.89, 2.42, 2.31, 2.09, 1.68, 1.49, 1.54, 0.91])  # Tumor volumes (cm³)

def tumor_model(t, S_max, k, t_0, lambda_):
    return (S_max / (1 + np.exp(-k * (t - t_0)))) * np.exp(-lambda_ * t)

# Initial guess for the parameters: S_max = 5 (estimated max size), k = 0.1, t_0 = 20, lambda_ = 0.05 (decay rate)
p0 = [5, 0.1, 20, 0.05]  # Initial guesses for the parameters

params, covariance = curve_fit(tumor_model, time_points_in_hours, tumor_volumes, p0)

S_max_fit, k_fit, t_0_fit, lambda_fit = params

plt.plot(time_points_in_hours, tumor_volumes, 'o', label='Observed Tumor Volumes')
plt.plot(time_points_in_hours, tumor_model(time_points_in_hours, *params), 'r-', label=f'Fitted Curve (λ={lambda_fit:.3f})')
plt.xlabel('Time (hours)')
plt.ylabel('Tumor Volume (cm³)')
plt.legend()
plt.show()

print(f"Estimated decay rate λ: {lambda_fit:.3f} hour^-1")
print(f"Max tumor size (S_max): {S_max_fit:.3f} cm³")
print(f"Estimated growth rate (k): {k_fit:.3f}")
print(f"Estimated inflection time (t_0): {t_0_fit:.3f} hours")

import numpy as np

# Tumor volume model for growth and decay
def tumor_model(t, S, S_max, k, lambda_):
    return (S / (1 + np.exp(-k * (t)))) * np.exp(-lambda_ * t)

# Objective function to evaluate the final tumor size
def objective_function(timing, S_max, k, t_0, lambda_):
    t_mev = timing[0]
    t_vsv = timing[1]

    # Initial tumor size at t = 0
    tumor_size = 2.47

    mev_effect = 0.05  # 5% tumor size reduction
    vsv_effect = 0.13   # 13% tumor size reduction

    # Tumor size just before and after MeV injection
    if t_mev <= 59:
        tumor_size = tumor_model(t_mev, tumor_size, S_max, k, lambda_)
        tumor_size *= (1 - mev_effect)

    # Tumor size just before and after VSV injection
    if t_vsv <= 59:
        tumor_size = tumor_model(t_vsv, tumor_size, S_max, k, lambda_)
        tumor_size *= (1 - vsv_effect)

    # Tumor size at Day 59
    tumor_size = tumor_model(59, tumor_size, S_max, k, lambda_)
    return tumor_size

# Parameters
S_max = 5.0
k = 0.1
lambda_ = 0.001
t_0 = 20

# Grid search over possible MeV and VSV timings
mev_times = np.array([1, 4, 8, 12, 15, 19])
vsv_times = np.array([26, 43, 50])

all_combinations = []

for mev in mev_times:
    for vsv in vsv_times:
        timing = [mev, vsv]
        final_tumor_size = objective_function(timing, S_max, k, t_0, lambda_)
        all_combinations.append((mev, vsv, final_tumor_size))

# Sort by final tumor size
all_combinations = sorted(all_combinations, key=lambda x: x[2])

# Print all results
print("MeV Timing | VSV Timing | Final Tumor Size (cm³)")
print("-----------------------------------------------")
for combination in all_combinations:
    print(f"Day {combination[0]:<9} | Day {combination[1]:<9} | {combination[2]:.3f}")

# Optimal timing
optimal_timing = all_combinations[0]
print("\nOptimal Timing:")
print(f"MeV Timing: Day {optimal_timing[0]}, VSV Timing: Day {optimal_timing[1]}")
print(f"Final Tumor Size: {optimal_timing[2]:.3f} cm³")

import numpy as np
import random
import matplotlib.pyplot as plt

def tumor_model(t_mev, t_vsv, d_mev, d_vsv, S0=2.61):
    t_mev_hours = t_mev * 24
    t_vsv_hours = t_vsv * 24

    # decay for MeV and VSV
    decay_mev = d_mev * np.exp(-0.1 * (t_mev_hours - 1))  # MeV decay effect
    decay_vsv = d_vsv * np.exp(-0.1 * (t_vsv_hours - t_mev_hours))  # VSV decay effect
    final_tumor_size = S0 - decay_mev - decay_vsv
    return max(final_tumor_size, 0)  # Tumor size should not go below 0

population_size = 50
generations = 100
mutation_rate = 0.1
t_mev_range = (1, 19)  # MeV timing range (Day 1 to Day 19)
t_vsv_range = (26, 50)  # VSV timing range (Day 26 to Day 50)
d_mev_range = (5.5, 7.8)  # MeV dosage range (logCCID50)
d_vsv_range = (7.9, 9.0)  # VSV dosage range (logCCID50)

# Fix the timings as provided
t_mev_fixed = 1  # MeV at Day 1
t_vsv_fixed = 26  # VSV at Day 26

# Initialize population
def initialize_population():
    population = []
    for _ in range(population_size):
        d_mev = random.uniform(*d_mev_range)
        d_vsv = random.uniform(*d_vsv_range)
        population.append([t_mev_fixed, t_vsv_fixed, d_mev, d_vsv])
    return population

def fitness_function(individual):
    t_mev, t_vsv, d_mev, d_vsv = individual
    return tumor_model(t_mev, t_vsv, d_mev, d_vsv)

def select_parents(population, fitness):
    indices = np.random.choice(range(len(population)), size=2, p=fitness/fitness.sum())
    return population[indices[0]], population[indices[1]]

def crossover(parent1, parent2):
    child = parent1[:2] + parent2[2:]
    return child

def mutate(individual):
    if random.random() < mutation_rate:
        individual[2] = random.uniform(*d_mev_range)  # Mutate MeV dosage
    if random.random() < mutation_rate:
        individual[3] = random.uniform(*d_vsv_range)  # Mutate VSV dosage
    return individual

# Genetic algorithm to optimize tumor size
def genetic_algorithm():
    population = initialize_population()
    for generation in range(generations):
        fitness = np.array([1 / (1 + fitness_function(ind)) for ind in population])
        new_population = []
        for _ in range(population_size // 2):
            parent1, parent2 = select_parents(population, fitness)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])
        population = new_population
        best_individual = population[np.argmax(fitness)]
        best_tumor_size = fitness_function(best_individual)
        print(f"Generation {generation}: Best Tumor Size = {best_tumor_size:.3f}, Best Individual = {best_individual}")
    return best_individual

optimal_solution = genetic_algorithm()
print(f"Optimal Solution: {optimal_solution}")

best_tumor_sizes = [1.876, 1.963, 1.984, 1.856, 1.856, 1.880, 2.015, 1.994, 2.015]
best_dosages = [
    [7.323, 7.912], [6.456, 8.465], [6.242, 8.708], [7.517, 8.470],
    [7.517, 8.470], [7.283, 8.708], [5.930, 8.638], [6.146, 8.201], [5.930, 8.638]
]

mev_dosages = [dosage[0] for dosage in best_dosages]
vsv_dosages = [dosage[1] for dosage in best_dosages]

plt.figure(figsize=(10, 6))
plt.plot(best_tumor_sizes, marker='o', linestyle='-', color='b')
plt.title('Tumor Size Evolution Over Generations')
plt.xlabel('Generation')
plt.ylabel('Best Tumor Size (cm³)')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(mev_dosages, marker='o', linestyle='-', color='r', label='MeV Dosage')
plt.title('MeV Dosage Evolution Over Generations')
plt.xlabel('Generation')
plt.ylabel('MeV Dosage (logCCID50)')
plt.grid(True)
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(vsv_dosages, marker='o', linestyle='-', color='g', label='VSV Dosage')
plt.title('VSV Dosage Evolution Over Generations')
plt.xlabel('Generation')
plt.ylabel('VSV Dosage (logCCID50)')
plt.grid(True)
plt.legend()
plt.show()
