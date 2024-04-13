import random


def generate_data():
    task_count = int(input("Enter the number of tasks: "))
    processor_count = int(input("Enter the number of processors: "))
    task_range = int(input("Enter the maximum task range (1 to N): "))

    tasks = [random.randint(1, task_range) for _ in range(task_count)]

    with open('data.txt', 'w') as file:
        file.write(f"{processor_count}\n")
        file.write(f"{task_count}\n")
        for task in tasks:
            file.write(f"{task}\n")


def main():
    generate_data()


if __name__ == "__main__":
    main()
