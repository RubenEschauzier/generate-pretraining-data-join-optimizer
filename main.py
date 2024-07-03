import os

from src.random_query_generation.main_generation import generate


def main(graph_location, task_type, save_location):
    if task_type == "generation":
        generate(graph_location, save_location)


if __name__ == "__main__":
    project_root = os.getcwd()
    instantiation_benchmark_location = r"C:\Users\ruben\benchmarks\dataset.nt"

    randomly_generated_queries_location = os.path.join(project_root, "output", "randomly_generated_queries",
                                                       "queries_complex.txt")

    generate(instantiation_benchmark_location, randomly_generated_queries_location)
    pass
