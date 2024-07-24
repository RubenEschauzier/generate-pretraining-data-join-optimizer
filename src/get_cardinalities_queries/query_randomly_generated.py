import json
import os
import pickle
from os import listdir
from os.path import join, isfile

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


def load_queries_watdiv(location):
    files = [f for f in listdir(location) if isfile(join(location, f))]

    queries = {}
    for file in files:
        with open(join(location, file), 'r') as f:
            queries[file] = f.read().strip().split('\n\n')

    return queries


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
            f.write(query)

    with open(save_location + "/cardinalities.txt", 'w') as f:
        for card in full_cardinalities:
            f.write(str(card))


def main_watdiv(endpoint, graph_uri, queries_location, save_location):
    query_per_template = load_queries_watdiv(queries_location)
    wrapped_endpoint = wrapper(endpoint, graph_uri)

    cardinality_per_template = {}
    processed_query_per_template = {}
    for (key, value) in query_per_template.items():
        _, cardinalities = execute_array_of_queries(value, wrapped_endpoint)
        cardinality_per_template[key.split('.')[0]] = cardinalities
        processed_query_per_template[key.split('.')[0]] = value

    with open(os.path.join(save_location, "template_cardinalities"), 'wb') as f:
        pickle.dump(cardinality_per_template, f)

    with open(os.path.join(save_location, "template_queries"), 'wb') as f:
        pickle.dump(processed_query_per_template, f)


if __name__ == "__main__":
    engine_endpoint = "http://localhost:8890/sparql"
    graph_uri_endpoint = "http://localhost:8890/watdiv-default-instantiation"

    project_root = os.path.join(os.getcwd(), "..", "..")

    randomly_generated_queries_location = os.path.join(project_root, "output", "randomly_generated_queries",
                                                       "queries_generated_large_less_empty")

    path_location = randomly_generated_queries_location + "_path.txt"
    star_location = randomly_generated_queries_location + "_star.txt"
    complex_location = randomly_generated_queries_location + "_complex.txt"

    random_dataset_save_location = os.path.join(project_root, "output", "randomly_generated_train_dataset")

    save_location_path = os.path.join(random_dataset_save_location, "path")
    save_location_star = os.path.join(random_dataset_save_location, "star")
    save_location_complex = os.path.join(random_dataset_save_location, "complex")

    random_dataset_ckp_location = os.path.join(random_dataset_save_location, "ckp")
    # TODO: Make directory ckp if it doesnt exist
    ckp_path = os.path.join(save_location_path, "ckp")
    ckp_star = os.path.join(save_location_star, "ckp")
    ckp_complex = os.path.join(save_location_complex, "ckp")

    watdiv_queries_location = os.path.join(project_root, "input", "watdiv_queries")
    watdiv_output_location = os.path.join(project_root, "output", "watdiv_query_cardinalities")

    # main_watdiv(engine_endpoint,
    #             graph_uri_endpoint,
    #             watdiv_queries_location,
    #             watdiv_output_location
    #             )
    print("Query Path")
    main(engine_endpoint,
         graph_uri_endpoint,
         path_location,
         random_dataset_save_location,
         parallel=True,
         n_proc=8,
         ckp_location=ckp_path
         )
    print("Query Star")
    main(engine_endpoint,
         graph_uri_endpoint,
         star_location,
         random_dataset_save_location,
         parallel=True,
         n_proc=8,
         ckp_location=ckp_star
         )
    print("Query Complex")
    main(engine_endpoint,
         graph_uri_endpoint,
         complex_location,
         random_dataset_save_location,
         parallel=True,
         n_proc=8,
         ckp_location=ckp_complex
         )
