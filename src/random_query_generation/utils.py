import math

import numpy as np
import random
import re
import rdflib.term
from rdflib.compare import to_isomorphic
from rdflib.graph import Graph
from sklearn.preprocessing import minmax_scale


class WrapperIsomorphicGraphFixedHashing:
    def __init__(self, graph):
        self.iso_graph = graph

    def __hash__(self):
        return self.iso_graph.internal_hash()

    def __eq__(self, other):
        return True


def filter_isomorphic_queries(queries):
    accepted_iso_set = set()
    accepted_queries = []

    for query in queries:
        iso = to_isomorphic(convert_query_to_graph(query))
        if WrapperIsomorphicGraphFixedHashing(iso) in accepted_iso_set:
            continue
        accepted_iso_set.add(WrapperIsomorphicGraphFixedHashing(iso))
        accepted_queries.append(query)
    print("FILTERING")
    print(len(queries))
    print(len(accepted_queries))
    return accepted_queries


def convert_query_to_graph(query):
    g = Graph()
    bgp = re.search('\{(.*?)\}', query).group().replace("}", "").replace("{", "")
    tps = bgp.split(" . ")[:-1]
    for tp in tps:
        spo = tp.strip().split(' ')
        triple_to_add = [rdflib.term.BNode(spo[0]), rdflib.term.URIRef(spo[1][1:-1])]
        if spo[2][0] == "?":
            triple_to_add.append(rdflib.term.BNode(spo[2]))
        else:
            triple_to_add.append(rdflib.term.Literal(spo[2]))
        g.add((triple_to_add[0], triple_to_add[1], triple_to_add[2]))
    return g


def get_all_subject(g):
    all_subj = set()
    for (subj, _, _) in g:
        all_subj.add(subj)
    return all_subj


def get_all_object(g):
    all_obj = set()
    for (_, _, obj) in g:
        all_obj.add(obj)
    return all_obj


def get_start_triple_walk(start_point, g):
    all_triples = list(g.triples((start_point, None, None)))
    chosen_index = random.randrange(len(all_triples))
    return all_triples[chosen_index]


def count_predicates_queries(queries):
    predicate_counts = dict()
    for query in queries:
        graph_of_query = convert_query_to_graph(query)
        for _, p, _ in graph_of_query:
            predicate_counts[p] = predicate_counts.get(p, 0) + 1

    return predicate_counts


def weighted_choice(weights):
    return random.choices(range(len(weights)), weights=weights)[0]


def choose_triple_weighted(all_triples, predicate_usage_counts):
    weights = []
    for triple in all_triples:
        predicate = triple[1]
        if predicate in predicate_usage_counts:
            weights.append(-(predicate_usage_counts[predicate] + 1))
        else:
            weights.append(-1)
    weights = np.array(weights)
    # Min-max normalization for numerical stability for very negative weights
    weights = minmax_scale(weights)

    # Choose according to weights
    probabilities = np.exp(weights) / sum(np.exp(weights))
    chosen_index = weighted_choice(probabilities)
    return chosen_index, predicate_usage_counts


def update_counts_predicates_used(all_triples, chosen_index, predicate_usage_counts):
    chosen_predicate = all_triples[chosen_index][1]

    # Update the counts
    if chosen_predicate in predicate_usage_counts:
        predicate_usage_counts[chosen_predicate] += 1
    else:
        predicate_usage_counts[chosen_predicate] = 1
    return predicate_usage_counts


# The idea is that if the set of predicates is equivalent to something already used and it is the same shape, the query
# is superfluous. Here the order does not matter
def track_equivalent_predicates(used_predicate_set, predicates):
    if predicates in used_predicate_set:
        return True
    used_predicate_set.add(tuple(predicates))
    return False


def filter_equivalent_queries(walk_query,
                              walk,
                              all_queries,
                              equivalent_predicates,
                              used_predicates_dict,
                              used_predicates,
                              non_variable_edges, ):
    if not equivalent_predicates:
        used_predicates_dict[used_predicates] = 1
        all_queries.append(walk_query)
    elif non_variable_edges > 0:
        # We scale the acceptance chance of a predicate combination that we've seen before with one literal with a
        # quasi-exponential function to prevent our queries primarily consisting of often occurring paths in the
        # graph
        n_usages = used_predicates_dict[used_predicates]
        accept_prob = 1 / math.exp(n_usages)
        r = random.uniform(0, 1)
        if r < accept_prob:
            all_queries.append(walk_query)
            used_predicates_dict[used_predicates] += 1
    return all_queries, used_predicates_dict


def generate_triple_string(triple, r, prob_non_variable_edge, term_to_variable_dict, object_centre):
    non_variable_edge = False
    if r > prob_non_variable_edge or object_centre:
        triple_string = "{} <{}> {} . ".format(
            term_to_variable_dict[triple[0]],
            triple[1],
            term_to_variable_dict[triple[2]]
        )
    elif object_centre:
        non_variable_edge = True
        if type(triple[0]) == rdflib.term.Literal:
            triple_string = "\"{}\" <{}> {} . ".format(
                triple[1],
                triple[2],
                term_to_variable_dict[triple[0]]
            )
        # If not literal, we need brackets around it
        else:
            triple_string = "<{}> <{}> {} . ".format(
                triple[1],
                triple[2],
                term_to_variable_dict[triple[0]]
            )
    else:
        non_variable_edge = True
        if type(triple[2]) == rdflib.term.Literal:
            triple_string = "{} <{}> \"{}\" . ".format(
                term_to_variable_dict[triple[0]],
                triple[1],
                triple[2]
            )
        # If not literal, we need brackets around it
        else:
            triple_string = "{} <{}> <{}> . ".format(
                term_to_variable_dict[triple[0]],
                triple[1],
                triple[2]
            )
    return triple_string, non_variable_edge
