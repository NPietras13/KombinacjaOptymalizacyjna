import time
import numpy as np
import random


# Funkcje pomocnicze (niezmienione)
def create_starting_individual(n_tasks):
    return np.arange(0, n_tasks)


def create_random_individual(n_tasks):
    return np.random.permutation(n_tasks)


def mutate(individual):
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


def crossover(parent1, parent2):
    size = len(parent1)

    # Wybieramy dwa punkty krzyżowania losowo
    start, end = sorted(random.sample(range(size), 2))

    # Inicjujemy dziecko z None
    child = [None] * size

    # Kopiujemy segment z pierwszego rodzica
    child[start:end + 1] = parent1[start:end + 1]

    # Wypełniamy pozostałe miejsca genami z drugiego rodzica w oryginalnej kolejności
    current_pos = 0
    for task in parent2:
        if task not in child:
            while child[current_pos] is not None:
                current_pos = (current_pos + 1) % size
            child[current_pos] = task

    return np.array(child)


def find_min(arr):
    if not arr:
        return -1
    min_value = min(arr)
    min_index = arr.index(min_value)
    return min_index


def find_max(arr):
    if not arr:
        return -1
    max_value = max(arr)
    max_index = arr.index(max_value)
    return max_index


def greedy_algorithm(individual, tasks, processor_count):
    processors = [0] * processor_count
    for task in individual:
        p_index = find_min(processors)
        processors[p_index] += tasks[task]
    return max(processors)


def tournament_selection(population, task_durations, n_processors, k=3):
    selected = []
    for _ in range(len(population)):
        tournament = random.sample(population, k)
        tournament_fitness = [greedy_algorithm(individual, task_durations, n_processors) for individual in tournament]
        winner_index = find_min(tournament_fitness)
        selected.append(tournament[winner_index])
    return selected


def elitism_selection(population, task_durations, n_processors, elite_size=2):
    fitness_values = [greedy_algorithm(individual, task_durations, n_processors) for individual in population]
    sorted_indices = np.argsort(fitness_values)
    selected = [population[idx] for idx in sorted_indices[:elite_size]]

    return selected


def genetic_algorithm_multi_processor(n_tasks, task_durations, max_time=180, population_size=100, n_processors=50,
                                      mutation_probability=0.1, crossover_probability=0.9, tournament_size=3,
                                      elite_size=2):
    generations = []
    population = [create_random_individual(n_tasks) for _ in range(population_size)]
    start_time = time.time()

    while time.time() - start_time < max_time:
        t_maxes = [greedy_algorithm(individual, task_durations, n_processors) for individual in population]
        new_population = elitism_selection(population, task_durations, n_processors, elite_size=elite_size)

        selected_population = tournament_selection(population, task_durations, n_processors, k=tournament_size)

        while len(new_population) < population_size:
            if random.random() < crossover_probability:

                parent1, parent2 = random.sample(selected_population, 2)

                child = crossover(parent1, parent2)

            else:
                child = random.choice(selected_population)

            if random.random() < mutation_probability:
                child = mutate(child)

            new_population.append(child)

        generations.append(population)
        population = new_population

    return generations


def load_data(filename="data.txt"):
    print(f"Wczytywanie danych z pliku {filename}")
    tasks = []
    with open(filename, "r") as file:
        processor_count = int(file.readline())
        task_count = int(file.readline())
        for line in file:
            tasks.append(int(line))
            if len(tasks) == task_count:
                break
    return task_count, tasks, processor_count


# Ładowanie danych
task_count, task_durations, n_processors = load_data()

# Pomiar czasu
start_time = time.time()

# Uruchomienie algorytmu genetycznego
generations = genetic_algorithm_multi_processor(task_count, task_durations, max_time=5, population_size=50,
                                                n_processors=n_processors, mutation_probability=0.10,
                                                crossover_probability=0.95, tournament_size=3, elite_size=2)

# Pomiar czasu zakończenia
end_time = time.time()
elapsed_time = end_time - start_time

# Ocena wyników
t_maxes = []
for population in generations:
    for individual in population:
        t_maxes.append(greedy_algorithm(individual, task_durations, n_processors))

print("Najlepszy czas: ", min(t_maxes))
print("Czas wykonania: ", elapsed_time, "sekund")
