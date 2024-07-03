import json
import os

from queryVirtuoso import wrapper, execute_array_of_queries


def load_watdiv(query_location):
    return json.load(open(query_location, "r"))
    pass


def main(endpoint, graph_uri, queries_location, dataset_save_location, ckp_location=None):
    queries = load_watdiv(queries_location)
    wrapped_endpoint = wrapper(endpoint, graph_uri)
    full_query_string, full_cardinalities = execute_array_of_queries(queries, wrapped_endpoint, ckp_location)

    with open(dataset_save_location + "/query_strings.json", "w") as f:
        json.dump(full_query_string, f)
    with open(dataset_save_location + "/query_cardinalities.json", "w") as f:
        json.dump(full_cardinalities, f)


if __name__ == "__main__":
    project_root = os.getcwd()

    watdiv_queries_location = os.path.join(project_root, "data_watdiv", "queries_watdiv",
                                           "cleaned_watdiv_queries.txt")
    save_location = os.path.join(project_root, "output", "validation_dataset_watdiv")
    dataset_ckp_location = os.path.join(save_location, "ckp")
    engine_endpoint = "http://localhost:8890/sparql"
    graph_uri_endpoint = "http://localhost:8890/watdiv-default-instantiation"

    query_location = "C:/Users/Administrator/projects/" \
                     "preprocess_data_rl_optimizer/data_watdiv/queries_watdiv/cleaned_watdiv_queries.txt"
    main(engine_endpoint,
         graph_uri_endpoint,
         query_location,
         save_location,
         dataset_ckp_location
         )
