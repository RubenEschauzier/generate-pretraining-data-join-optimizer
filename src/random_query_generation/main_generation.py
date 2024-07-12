import itertools
import os

import numpy as np
from rdflib import Graph
from src.random_query_generation.generate_complex_queries import generate_complex_queries
from src.random_query_generation.generate_path_queries import generate_path_queries
from src.random_query_generation.generate_star_queries import generate_star_queries
from src.random_query_generation.utils import filter_isomorphic_queries


def generate(dataset_location, save_location=None):
    instantiation_benchmark_location = r"C:\Users\ruben\benchmarks\dataset.nt"

    input_graph = Graph()
    input_graph.parse(dataset_location, format="nt")
    print("Generating complex randomly_generated_queries")
    complex_queries, pred_counts = generate_complex_queries(input_graph,
                                                            repeats=2,
                                                            min_size_walk=3,
                                                            max_size_walk=7,
                                                            p_literal=.5,
                                                            p_walk_corrupt=.02)
    print("Generating path randomly_generated_queries")
    generated_path_queries = generate_path_queries(input_graph,
                                                   repeats=5,
                                                   max_walk_size=5,
                                                   p_literal=.5,
                                                   p_walk_corrupt=.02)
    print("Generating star randomly_generated_queries")
    subject_stars, object_stars = generate_star_queries(input_graph,
                                                        repeats=10,
                                                        max_size_star=5,
                                                        p_literal=.25,
                                                        p_walk_corrupt=.02)

    all_queries = np.array(list(itertools.chain(subject_stars, object_stars, generated_path_queries, complex_queries)))
    filtered_queries = filter_isomorphic_queries(all_queries)
    np.random.shuffle(filtered_queries)

    if save_location:
        with open(save_location, "w") as f:
            for query in filtered_queries:
                f.write(query)
                f.write('\n')


if __name__ == "__main__":
    project_root = os.getcwd()

    randomly_generated_queries_location = os.path.join(project_root, "output", "randomly_generated_queries",
                                                       "queries_generated_large_less_empty.txt")

    dataset_location = r"C:\Users\ruben\benchmarks\dataset.nt"
    generate(dataset_location, randomly_generated_queries_location)
