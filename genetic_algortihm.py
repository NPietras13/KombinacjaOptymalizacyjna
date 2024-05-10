import numpy as np
import random
import itertools


# Funkcja celu: obliczenie "makespan" oraz harmonogram dla wielu procesorów
def multi_processor_schedule_and_makespan(schedule, task_durations, n_processors):
    """
    Oblicz harmonogram zadań na wielu procesorach oraz ich "makespan".
    """
    # Używamy listy, aby śledzić zadania przypisane do każdego procesora
    processor_schedules = [[] for _ in range(n_processors)]
    processor_loads = [0] * n_processors

    # Przypisanie zadań do procesorów
    for task in schedule:
        min_load_processor = processor_loads.index(min(processor_loads))
        processor_schedules[min_load_processor].append(task)
        processor_loads[min_load_processor] += task_durations[task]

    # Obliczenie "makespan"
    makespan = max(processor_loads)

    return processor_schedules, makespan


def create_multi_processor_individual(n_tasks):
    """
    Stwórz losowy harmonogram dla wielu procesorów.
    """
    return np.random.permutation(n_tasks)


def mutate(individual):
    """
    Mutacja polegająca na zamianie miejscami dwóch losowych zadań.
    """
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


def crossover(parent1, parent2):
    """
    Operacja krzyżowania Order Crossover (OX).
    """
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))
    child = [None] * size

    child[start:end + 1] = parent1[start:end + 1]

    current_pos = (end + 1) % size
    for task in parent2:
        if task not in child:
            while child[current_pos] is not None:
                current_pos = (current_pos + 1) % size
            child[current_pos] = task

    return np.array(child)


def selection(population, fitnesses):
    """
    Selekcja ruletkowa.
    """
    total_fitness = sum(fitnesses)
    pick = random.uniform(0, total_fitness)
    current = 0
    for idx, fitness in enumerate(fitnesses):
        current += fitness
        if current > pick:
            return population[idx]
    return population[-1]


# Główna funkcja algorytmu genetycznego
def genetic_algorithm_multi_processor(task_durations, num_generations=100, population_size=30, n_processors=3):
    n_tasks = len(task_durations)

    population = [create_multi_processor_individual(n_tasks) for _ in range(population_size)]

    for generation in range(num_generations):
        # Obliczanie fitness dla całej populacji
        fitnesses = [1 / multi_processor_schedule_and_makespan(individual, task_durations, n_processors)[1]
                     for individual in population]

        new_population = []

        while len(new_population) < population_size:
            parent1 = selection(population, fitnesses)
            parent2 = selection(population, fitnesses)

            child = crossover(parent1, parent2)

            if random.random() < 0.1:
                child = mutate(child)

            new_population.append(child)

        population = new_population

    # Znalezienie najlepszego harmonogramu
    final_fitnesses = [1 / multi_processor_schedule_and_makespan(individual, task_durations, n_processors)[1]
                       for individual in population]
    best_index = np.argmax(final_fitnesses)

    best_schedule = population[best_index]
    processor_schedules, best_makespan = multi_processor_schedule_and_makespan(best_schedule, task_durations,
                                                                               n_processors)

    return processor_schedules, best_makespan


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

# Przykład użycia:
task_count, task_durations, n_processors = load_data()

processor_schedules, best_makespan = genetic_algorithm_multi_processor(task_durations, num_generations=100,
                                                                       population_size=30, n_processors=n_processors)

print("Najlepszy harmonogram dla każdego procesora:")
for i, schedule in enumerate(processor_schedules):
    print(f"Procesor {i + 1}: {schedule}")

print("Całkowity czas ukończenia (makespan):", best_makespan)
