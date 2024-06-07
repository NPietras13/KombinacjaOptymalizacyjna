import time


# def find_max(arr):
#     if not arr:
#         return -1
#     max_value = max(arr)
#     max_index = arr.index(max_value)
#     return max_index

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


def find_min(arr):
    if not arr:
        return -1

    min_value = min(arr)
    min_index = arr.index(min_value)
    return min_index


def greedy_algorithm(tasks, processor_count):
    processors = [0] * processor_count
    print(tasks)
    for task in tasks:
        p_index = find_min(processors)
        processors[p_index] += task
        print(str(p_index) + ': ' + str(task))
    return processors


def main():
    task_count, tasks, processor_count = load_data()

    start_time = time.time()
    processors = greedy_algorithm(tasks, processor_count)
    elapsed_time = time.time() - start_time

    i = 0
    max_time = 0
    for processor in processors:
        if i == 0:
            max_time = i
        else:
            if processor > processors[max_time]:
                max_time = i
        print('P' + str(i+1) + ': ' + str(processor) )
        i += 1

    # print(f"Execution time for greedy_algorithm: {elapsed_time:.6f} seconds")

    print('Wynik: ' + str(processors[max_time]))


if __name__ == "__main__":
    main()
