import pandas
from rdflib import Graph
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from testVectorsPredicates import load_clean_data, get_all_queries
import re


def get_queries(file_loc_dataset, file_loc_vectors):
    emb, pred, pred_emb_dict = load_clean_data(file_loc_dataset, file_loc_vectors)
    star_in, star_out, chain, chain_reversed, sim = get_all_queries(pred_emb_dict, 2000)
    return star_in, star_out, chain, chain_reversed, sim, emb, pred, pred_emb_dict


def load_graph(file_location):
    rdf_graph = Graph()
    rdf_graph.parse(file_location, format='nt')
    return rdf_graph


def obtain_datapoints_query(g, queries, emb, pred_emb_dict):
    X = []
    y = []
    for query_trio in queries:
        features = []
        card1 = len(g.query(query_trio[0]))
        card2 = len(g.query(query_trio[1]))
        card3 = len(g.query(query_trio[2]))
        selectivity = card3 / (card1 + card2)

        features.append(card1)
        features.append(card2)

        pred_in_query_0 = get_predicates_query(g, query_trio[0])
        pred_in_query_1 = get_predicates_query(g, query_trio[1])
        features.extend(pred_emb_dict[pred_in_query_0])
        features.extend(pred_emb_dict[pred_in_query_1])

        X.append(features)
        y.append(selectivity)
    X = pandas.DataFrame(X, columns=["F{}".format(i) for i in range(len(X[0]))])
    X[['F0', 'F1']] = StandardScaler().fit_transform(X[['F0', 'F1']])
    return X, y


def cross_validate_svm(X, y):
    clf = SVR(kernel='linear', C=1)
    scores = cross_val_score(clf, X, y, cv=5)
    print(scores)
    pass


def get_predicates_query(g, query):
    found = re.findall(r"<(.*?)>", query)
    return found[0]


def run_predictor(file_loc_dataset, file_loc_vectors):
    g = load_graph("data_watdiv/dataset.nt")
    star_in, star_out, chain, chain_reversed, sim, emb, _, pred_emb_dict = \
        get_queries(file_loc_dataset, file_loc_vectors)
    star_X, star_y = obtain_datapoints_query(g, star_in, emb, pred_emb_dict)
    cross_validate_svm(star_X, star_y)


if __name__ == "__main__":
    run_predictor('data_watdiv/dataset.nt', 'data_watdiv/vectorsWatDiv/vectors_depth_2.txt')
