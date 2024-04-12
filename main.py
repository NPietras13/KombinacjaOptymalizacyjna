import random


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


def generate_data():
    task_count = 15
    tasks = [random.randint(1, 10) for _ in range(task_count)]
    processor_count = 3
    return task_count, tasks, processor_count


def main():
    task_count, tasks, processor_count = generate_data()
    # task_count, tasks, processor_count = load_data()

    processors = [0] * processor_count
    print(tasks)
    for task in tasks:
        p_index = find_min(processors)
        processors[p_index] += task

    i = 1
    for processor in processors:
        print('P' + str(i) + ': ' + str(processor))
        i += 1


if __name__ == "__main__":
    main()
