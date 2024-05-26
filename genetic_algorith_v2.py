import numpy as np
import random
def create_starting_individual(n_tasks):
    """
    Stwórz losowego osobnika
    """
    return np.random.permutation(n_tasks)


def mutate(individual):
    """
    Mutacja polegająca na zamianie miejscami dwóch losowych zadań.
    """
    idx1, idx2 = random.sample(range(len(individual)), 2)
    individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual


def crossover(parent2, parent1):
    """
    Operacja krzyżowania (OX).
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


def genetic_algorithm_multi_processor(n_tasks, task_durations, num_generations=20, population_size=20, n_processors=3):
    generations = []
    new_population = []
    t_maxes = []
    for generation in range(num_generations):
        if(len(new_population) <= 0):
            population = [create_starting_individual(n_tasks) for _ in range(2)]
        else:
            population = [create_starting_individual(n_tasks) for _ in range(1)]
            population.append(new_population[find_min(t_maxes)])
        new_population = population
        while len(new_population) < population_size:
            t_maxes = []
            for individual in population:
                t_maxes.append(greedy_algorithm(individual, task_durations, n_processors))
            temp = t_maxes.copy()
            print(t_maxes)
            if random.random() < 0.2:
                parent1 = population[find_min(temp)]
                parent2 = population[random.randint(0, len(population) - 1)]
            else:
                parent1 = population[find_min(temp)]
                print(min(temp))
                temp.pop(find_min(temp))
                parent2 = population[find_min(temp)]

            child = crossover(parent1, parent2)
            if random.random() < 0.2:
                child = mutate(child)

            new_population.append(child)
        generations.append(population)
        population = new_population

    # Znalezienie najlepszego harmonogramu

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


task_count, task_durations, n_processors = load_data()
generations = genetic_algorithm_multi_processor(task_count, task_durations, num_generations=100,
                                                                       population_size=2000, n_processors=n_processors)

t_maxes = []
for population in generations:
    for individual in population:
        t_maxes.append(greedy_algorithm(individual, task_durations, n_processors))

print(min(t_maxes))
