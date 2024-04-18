import itertools
import numpy as np
from rdflib import Graph
from src.random_query_generation.generate_complex_queries import generate_complex_queries
from src.random_query_generation.generate_path_queries import generate_path_queries
from src.random_query_generation.generate_star_queries import generate_star_queries
from src.random_query_generation.utils import filter_isomorphic_queries


def main(save_location=None):
    instantiation_benchmark_location = r"C:\Users\Administrator\projects\benchmarks\watdiv-dataset\dataset.nt"
    input_graph = Graph()
    input_graph.parse(instantiation_benchmark_location, format="nt")
    print("Generating complex queries")
    complex_queries = generate_complex_queries(input_graph, 1, 3, 7, .3)
    print("Generating path queries")
    generated_path_queries = generate_path_queries(input_graph, 2, 5, .25)
    print("Generating star queries")
    subject_stars, object_stars = generate_star_queries(input_graph, 2, 5, .25)

    all_queries = np.array(list(itertools.chain(subject_stars, object_stars, generated_path_queries, complex_queries)))
    filtered_queries = filter_isomorphic_queries(all_queries)
    np.random.shuffle(filtered_queries)

    if save_location:
        np.save(save_location, filtered_queries)


if __name__ == "__main__":
    main("output/queries/queries_complex.npy")
    # main()
