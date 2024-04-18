import json

from src.train_dataset_generation.queryVirtuoso import wrapper, execute_array_of_queries


def load_watdiv(query_location):
    return json.load(open(query_location, "r"))
    pass


def main(endpoint, graph_uri, queries_location, dataset_save_location, ckp_location=None):
    queries = load_watdiv(queries_location)
    wrapped_endpoint = wrapper(endpoint, graph_uri)
    full_query_string, full_cardinalities = execute_array_of_queries(queries, wrapped_endpoint, ckp_location,
                                                                     queryRandom=False)

    with open(dataset_save_location + "/query_strings.json", "w") as f:
        json.dump(full_query_string, f)
    with open(dataset_save_location + "/query_cardinalities.json", "w") as f:
        json.dump(full_cardinalities, f)


if __name__ == "__main__":
    query_location = "C:/Users/Administrator/projects/" \
                     "preprocess_data_rl_optimizer/dataWatDiv/queriesWatDiv/cleanedQueries.txt"
    main("http://localhost:8890/sparql",
         "http://localhost:8890/watdiv-default-instantiation",
         query_location,
         "C:/Users/Administrator/projects/preprocess_data_rl_optimizer/output/validation_dataset_watdiv",
         "C:/Users/Administrator/projects/preprocess_data_rl_optimizer/output/validation_dataset_watdiv/chkp"
         )
