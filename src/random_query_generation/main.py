import itertools
import numpy as np
from rdflib import Graph
from src.random_query_generation.generate_complex_queries import generate_complex_queries
from src.random_query_generation.generate_path_queries import generate_path_queries
from src.random_query_generation.generate_star_queries import generate_star_queries


def main(save_location=None):
    instantiation_benchmark_location = r"C:\Users\Administrator\projects\benchmarks\watdiv-dataset\dataset.nt"
    input_graph = Graph()
    input_graph.parse(instantiation_benchmark_location, format="nt")
    generate_complex_queries(input_graph, 1, 3, 4, .8)
    generated_path_queries = generate_path_queries(input_graph, 1, 5, .25)
    subject_stars, object_stars = generate_star_queries(input_graph, 1, 5, .25)

    all_queries = np.array(list(itertools.chain(subject_stars, object_stars, generated_path_queries)))
    np.random.shuffle(all_queries)

    if save_location:
        np.save(save_location, all_queries)


if __name__ == "__main__":
    # main("output/queries/queries_star_path_iso.npy")
    main()
