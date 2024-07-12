import json
import os

import numpy as np

from queryVirtuoso import execute_array_of_queries, wrapper, query_parallel


# See following for starting dockerfile https://hub.docker.com/r/openlink/virtuoso-opensource-7

def load_queries(query_location):
    return np.load(query_location)


def load_queries_text(query_location):
    queries = []
    with open(query_location, 'r') as f:
        for line in f.readlines():
            queries.append(line)
    return np.array(queries)


def main(endpoint, graph_uri, queries_location, save_location, parallel=False, n_proc=2, ckp_location=None):
    queries = load_queries_text(queries_location)
    wrapped_endpoint = wrapper(endpoint, graph_uri)

    if parallel:
        full_query_string, full_cardinalities = query_parallel(n_proc, queries, wrapped_endpoint, ckp_location)
    else:
        full_query_string, full_cardinalities = execute_array_of_queries(queries, wrapped_endpoint, ckp_location)

    with open(save_location + "/query_strings.json", "w") as f:
        json.dump(full_query_string, f)
    with open(save_location + "/query_cardinalities.json", "w") as f:
        json.dump(full_cardinalities, f)

    with open(save_location + "/queries.txt", 'w') as f:
        for query in full_query_string:
            f.write(query + "\n")

    with open(save_location + "/cardinalities.txt", 'w') as f:
        for card in full_cardinalities:
            f.write(str(card) + "\n")


if __name__ == "__main__":
    project_root = os.path.join(os.getcwd(), "..", "..")
    randomly_generated_queries_location = os.path.join(project_root, "output", "randomly_generated_queries",
                                                       "queries_generated_large_less_empty.txt")
    dataset_save_location = os.path.join(project_root, "output", "randomly_generated_train_dataset")
    dataset_ckp_location = os.path.join(dataset_save_location, "ckp")
    engine_endpoint = "http://localhost:8890/sparql"
    graph_uri_endpoint = "http://localhost:8890/watdiv-default-instantiation"

    main(engine_endpoint,
         graph_uri_endpoint,
         randomly_generated_queries_location,
         dataset_save_location,
         parallel=True,
         n_proc=8,
         ckp_location=dataset_ckp_location
         )
