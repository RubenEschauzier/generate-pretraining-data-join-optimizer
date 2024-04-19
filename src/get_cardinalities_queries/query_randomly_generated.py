import json
import os

import numpy as np

from queryVirtuoso import execute_array_of_queries, wrapper


def load_queries(query_location):
    return np.load(query_location)


def main(endpoint, graph_uri, queries_location, dataset_save_location, ckp_location=None):
    queries = load_queries(queries_location)
    wrapped_endpoint = wrapper(endpoint, graph_uri)
    full_query_string, full_cardinalities = execute_array_of_queries(queries, wrapped_endpoint, ckp_location)

    with open(dataset_save_location + "/query_strings.json", "w") as f:
        json.dump(full_query_string, f)
    with open(dataset_save_location + "/query_cardinalities.json", "w") as f:
        json.dump(full_cardinalities, f)


if __name__ == "__main__":
    project_root = os.path.join(os.getcwd(), "..", "..")

    randomly_generated_queries_location = os.path.join(project_root, "output", "randomly_generated_queries",
                                                       "queries_complex.npy")
    dataset_save_location = os.path.join(project_root, "output", "randomly_generated_train_dataset")
    dataset_ckp_location = os.path.join(dataset_save_location, "ckp")
    engine_endpoint = "http://localhost:8890/sparql"
    graph_uri_endpoint = "http://localhost:8890/watdiv-default-instantiation"

    main(engine_endpoint,
         graph_uri_endpoint,
         randomly_generated_queries_location,
         dataset_save_location,
         dataset_ckp_location
         )
