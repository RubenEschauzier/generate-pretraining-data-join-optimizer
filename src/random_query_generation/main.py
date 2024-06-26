import itertools
import os

import numpy as np
from rdflib import Graph
from generate_complex_queries import generate_complex_queries
from generate_path_queries import generate_path_queries
from generate_star_queries import generate_star_queries
from utils import filter_isomorphic_queries


def main(save_location=None):
    instantiation_benchmark_location = r"C:\Users\Administrator\projects\benchmarks\watdiv-dataset\dataset.nt"
    input_graph = Graph()
    input_graph.parse(instantiation_benchmark_location, format="nt")
    print("Generating complex randomly_generated_queries")
    complex_queries = generate_complex_queries(input_graph, 100, 3, 7, .3)
    print("Generating path randomly_generated_queries")
    generated_path_queries = generate_path_queries(input_graph, 15, 5, .25)
    print("Generating star randomly_generated_queries")
    subject_stars, object_stars = generate_star_queries(input_graph, 15, 5, .25)

    all_queries = np.array(list(itertools.chain(subject_stars, object_stars, generated_path_queries, complex_queries)))
    filtered_queries = filter_isomorphic_queries(all_queries)
    np.random.shuffle(filtered_queries)

    if save_location:
        np.save(save_location, filtered_queries)


if __name__ == "__main__":
    project_root = os.getcwd()

    randomly_generated_queries_location = os.path.join(project_root, "output", "randomly_generated_queries",
                                                       "queries_complex.npy")

    main(randomly_generated_queries_location)
