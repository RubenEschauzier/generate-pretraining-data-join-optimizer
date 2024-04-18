import json
import numpy as np

from src.train_dataset_generation.queryVirtuoso import execute_array_of_queries, wrapper


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
    main("http://localhost:8890/sparql",
         "http://localhost:8890/watdiv-default-instantiation",
         "C:/Users/Administrator/projects/preprocess_data_rl_optimizer/output/queries/queries_star_path.npy",
         "C:/Users/Administrator/projects/preprocess_data_rl_optimizer/output/pretrain_dataset_full",
         "C:/Users/Administrator/projects/preprocess_data_rl_optimizer/output/pretrain_dataset_full/chkp"
         )
