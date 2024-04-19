import math

import numpy as np
import random

import pandas as pd
from rdflib.graph import Graph
from numpy import dot
from numpy.linalg import norm
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.manifold import TSNE
import plotly.express as px


def load_rdf_graph_predicates(file_location):
    """

    :param file_location: Location of input RDF graph
    :return: dictionary with rdf entities mapped to vectors, initialised as 0.
    """
    predicates_vectors = []
    predicates_vectors_dict = {}

    rdf_graph = Graph()
    rdf_graph.parse(file_location, format='nt')

    for (subj, pred, obj) in rdf_graph:
        predicates_vectors.append(str(pred))
        predicates_vectors_dict[str(pred)] = 0
    return predicates_vectors_dict


def load_rdf_embeddings(file_location):
    """
    Loads all rdf embeddings
    :param file_location: Location of text file with embeddings, with RDF term seperated from embeddings by
    [sep] and the embedding elements seperated by ','.
    :return: List with RDF terms and their associated embeddings ordered as: embeddings, predicates
    """
    vectors = []
    predicates = []

    with open(file_location) as f:
        for embedding in f.readlines():
            predicate = embedding.split('[sep]')[0]
            vector = np.array(embedding.split('[sep]')[1].strip().split(' '), dtype=float)

            vectors.append(vector)
            predicates.append(predicate)

    return vectors, predicates


def remove_non_predicate_vectors(pred_dict, vectors, predicates):
    clean_vectors = []
    clean_predicates = []
    for i in range(len(predicates)):
        if predicates[i] in pred_dict:
            clean_vectors.append(vectors[i])
            clean_predicates.append(predicates[i])
    return clean_vectors, clean_predicates


def populate_pred_embedding_dict(pred_dict, vectors, predicates):
    for (vector, predicate) in zip(vectors, predicates):
        pred_dict[predicate] = vector
    return pred_dict


def plot_embeddings_tsne(embeddings, predicates):
    dim_reduced_emb = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(embeddings)
    fig = px.scatter(x=dim_reduced_emb[:, 0], y=dim_reduced_emb[:, 1], text=predicates)
    fig.show()


def sample_predicate_pairs(num_samples, pred_vec_dict):
    to_evaluate = []
    cosine_similarities = []
    print("Sampling {} pairs".format(min(num_samples, math.factorial(len(pred_vec_dict.keys())))))
    for i in range(min(num_samples, math.factorial(len(pred_vec_dict.keys())))):
        predicates_sampled = random.sample(list(pred_vec_dict), 2)
        a = np.array(pred_vec_dict.get(predicates_sampled[0]))
        b = np.array(pred_vec_dict.get(predicates_sampled[1]))
        # If we have a null vector as predicate embedding cosine similarity has no meaning and is nan
        if norm(a) != 0 and norm(b) != 0:
            similarity = dot(a, b) / (norm(a) * norm(b))
            to_evaluate.append(predicates_sampled)
            cosine_similarities.append(similarity)
    return to_evaluate, cosine_similarities


def generate_test_star_queries(combination):
    queries_star_in = []
    queries_star_out = []

    for predicate_pair in combination:
        query_string_1 = "SELECT ?x ?y WHERE {?x <" + str(predicate_pair[0]) + "> ?y}"
        query_string_2 = "SELECT ?x ?y WHERE {?x <" + str(predicate_pair[1]) + "> ?y}"
        query_join_star_in = "SELECT ?x ?y ?z WHERE {?x <" + str(predicate_pair[0]) + "> ?y . ?x <" + str(
            predicate_pair[1]) + "> ?z}"
        query_join_star_out = "SELECT ?x ?y ?z WHERE {?x <" + str(predicate_pair[0]) + "> ?y . ?z <" + str(
            predicate_pair[1]) + "> ?y}"

        queries_star_in.append([query_string_1, query_string_2, query_join_star_in])
        queries_star_out.append([query_string_1, query_string_2, query_join_star_out])
    return queries_star_in, queries_star_out


def generate_test_chain_queries(combinations):
    queries_chain = []
    queries_chain_reversed = []

    for predicate_pair in combinations:
        string_single_1 = "SELECT ?x ?y WHERE {?x <" + str(predicate_pair[0]) + "> ?y}"
        string_single_2 = "SELECT ?x ?y WHERE {?x <" + str(predicate_pair[1]) + "> ?y}"
        string_join = "SELECT ?x ?y ?z WHERE {?x <" + str(predicate_pair[0]) + "> ?y . ?y <" + str(
            predicate_pair[1]) + "> ?z}"
        string_join_reversed = "SELECT ?x ?y ?z WHERE {?x <" + str(predicate_pair[1]) + "> ?y . ?y <" + str(
            predicate_pair[0]) + "> ?z}"
        queries_chain.append([string_single_1, string_single_2, string_join])
        queries_chain_reversed.append([string_single_1, string_single_2, string_join_reversed])
    return queries_chain, queries_chain_reversed


def load_clean_data(file_location_graph, file_location_vectors):
    # Load required data
    pred_vec_dict = load_rdf_graph_predicates(file_location_graph)
    embeddings, rdf_terms = load_rdf_embeddings(file_location_vectors)

    # Remove non predicates from obtained vectors and populate the dictionary
    pred_embeddings, predicates = remove_non_predicate_vectors(pred_vec_dict, embeddings, rdf_terms)
    pred_vec_dict = populate_pred_embedding_dict(pred_vec_dict, pred_embeddings, predicates)
    return pred_embeddings, predicates, pred_vec_dict


def get_all_queries(pred_vec_dict, n_samples):
    # Sample some pairs
    pairs, similarities = sample_predicate_pairs(n_samples, pred_vec_dict)

    star_queries_in, star_queries_out = generate_test_star_queries(pairs)
    chain_queries, chain_queries_reversed = generate_test_chain_queries(pairs)
    return star_queries_in, star_queries_out, chain_queries, chain_queries_reversed, similarities


def get_selectivity_query_combination(query, rdf_graph):
    n_results_1 = len(rdf_graph.query(query[0]))
    n_results_2 = len(rdf_graph.query(query[1]))
    n_results_3 = len(rdf_graph.query(query[2]))
    selectivity = n_results_3 / (n_results_1 + n_results_2)
    return selectivity


def execute_queries(file_location, star_queries_in, star_queries_out, chain_queries, chain_queries_reversed):
    rdf_graph = Graph()
    rdf_graph.parse(file_location, format='nt')

    selectivity_s_in = []
    selectivity_s_out = []
    selectivity_c = []
    selectivity_c_rev = []

    for (s_in, s_out, c, c_rev) in zip(star_queries_in, star_queries_out, chain_queries, chain_queries_reversed):
        selectivity_s_in.append(get_selectivity_query_combination(s_in, rdf_graph))
        selectivity_s_out.append(get_selectivity_query_combination(s_out, rdf_graph))
        selectivity_c.append(get_selectivity_query_combination(c, rdf_graph))
        selectivity_c_rev.append(get_selectivity_query_combination(c_rev, rdf_graph))

    return selectivity_s_in, selectivity_s_out, selectivity_c, selectivity_c_rev


def regression_analysis_selectivities(X, y):
    if len(X.shape) == 1:
        X = X.reshape(-1, 1)
    if len(y.shape) == 1:
        y = y.reshape(-1, 1)
    X = sm.add_constant(X)
    X = X.transpose()
    df_dict = {'constant': X[0, :], 'similarity': X[1, :], 'y': y.squeeze()}
    df = pd.DataFrame(data = df_dict)
    X_const = sm.add_constant(df.similarity)

    model = sm.OLS(df.y, X_const, hasconst=False)
    res = model.fit()
    print(res.summary())


def regression_all_queries(sim, sel_s_in, sel_s_out, sel_c, sel_c_rev):
    print("Regression results incoming star randomly_generated_queries")
    regression_analysis_selectivities(sim, sel_s_in)
    print("Regression results outgoing star randomly_generated_queries")
    regression_analysis_selectivities(sim, sel_s_out)
    print("Regression results chain randomly_generated_queries")
    regression_analysis_selectivities(sim, sel_c)
    print("Regression results inversed chain randomly_generated_queries")
    regression_analysis_selectivities(sim, sel_c_rev)

if __name__ == "__main__":
    emb, pred, pred_emb_dict = load_clean_data('data_watdiv/dataset.nt', 'data_watdiv/vectorsWatDiv/vectors_depth_2.txt')
    star_in, star_out, chain, chain_reversed, sim = get_all_queries(pred_emb_dict, 2000)
    sel_s_in, sel_s_out, sel_c, sel_c_rev = execute_queries('data_watdiv/dataset.nt', star_in, star_out, chain,
                                                            chain_reversed)
    regression_all_queries(np.array(sim), np.array(sel_s_in), np.array(sel_s_out),
                           np.array(sel_c), np.array(sel_c_rev))
# regression_analysis_selectivities(np.array(sim), np.array(sel_s_in))

# queries_low = []

# queries_high = []
# for low_predicate in low_predicates:
#     stringSingle1 = "SELECT ?x ?y WHERE {?x <" + str(low_predicate[0]) + "> ?y}"
#     stringSingle2 = "SELECT ?x ?y WHERE {?x <" + str(low_predicate[1]) + "> ?y}"
#     stringJoin = "SELECT ?x ?y ?z WHERE {?x <" + str(low_predicate[0]) + "> ?y . ?x <" + str(
#         low_predicate[1]) + "> ?z}"
#     queries_low.append([stringSingle1, stringSingle2, stringJoin])
#
# for high_predicate in high_predicates:
#     stringSingle1 = "SELECT ?x ?y WHERE {?x <" + str(high_predicate[0]) + "> ?y}"
#     stringSingle2 = "SELECT ?x ?y WHERE {?x <" + str(high_predicate[1]) + "> ?y}"
#     stringJoin = "SELECT ?x ?y ?z WHERE {?x <" + str(high_predicate[0]) + "> ?y . ?x <" + str(
#         high_predicate[1]) + "> ?z}"
#     queries_high.append([stringSingle1, stringSingle2, stringJoin])
#
# for i, query_low in enumerate(queries_low):
#     with open("testQueriesRDF2Vec/LowQueries/query{}".format(i), 'a') as f:
#         for query in query_low:
#             f.write(query)
#             f.write('[sep]')
# print("Done!")
# for i, query_high in enumerate(queries_high):
#     with open("testQueriesRDF2Vec/HighQueries/query{}".format(i), 'a') as f:
#         for query in query_high:
#             print(query)
#             f.write(query)
#             f.write('[sep]')
# print(queries_low)
# queries_low = []
# queries_high = []
# for low_predicate in low_predicates:
#     stringSingle1 = "SELECT ?x ?y WHERE {?x <" + str(low_predicate[0]) + "> ?y}"
#     stringSingle2 = "SELECT ?x ?y WHERE {?x <" + str(low_predicate[1]) + "> ?y}"
#     stringJoin = "SELECT ?x ?y ?z WHERE {?x <" + str(low_predicate[0]) + "> ?y . ?y <" + str(
#         low_predicate[1]) + "> ?z}"
#     queries_low.append([stringSingle1, stringSingle2, stringJoin])
#
# for high_predicate in high_predicates:
#     stringSingle1 = "SELECT ?x ?y WHERE {?x <" + str(high_predicate[0]) + "> ?y}"
#     stringSingle2 = "SELECT ?x ?y WHERE {?x <" + str(high_predicate[1]) + "> ?y}"
#     stringJoin = "SELECT ?x ?y ?z WHERE {?x <" + str(high_predicate[0]) + "> ?y . ?y <" + str(
#         high_predicate[1]) + "> ?z}"
#     queries_high.append([stringSingle1, stringSingle2, stringJoin])
#
# for i, query_low in enumerate(queries_low):
#     with open("testQueriesChainRDF2VecBerlin/LowQueries/query{}".format(i), 'a') as f:
#         for query in query_low:
#             f.write(query)
#             f.write('[sep]')
# for i, query_high in enumerate(queries_high):
#     with open("testQueriesChainRDF2VecBerlin/HighQueries/query{}".format(i), 'a') as f:
#         for query in query_high:
#             f.write(query)
#             f.write('[sep]')


# rdf_predicates = [rdflib.term.URIRef(x) for x in to_evaluate_predicates]
# triples_with_predicate_1 = list(g.triples((None, rdf_predicates[0], None)))
# triples_with_predicate_2 = list(g.triples((None, rdf_predicates[1], None)))
# triples_to_test1 = random.sample(triples_with_predicate_1, 5)
# triples_to_test2 = random.sample(triples_with_predicate_2, 5)
# print(triples_to_test1)
